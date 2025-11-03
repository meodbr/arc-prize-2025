from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np
import torch
import torch.nn.functional as F
import random

from arc_tartiflette.inference.solver import Solver
from arc_tartiflette.utils import load, constants
from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format
from arc_tartiflette.model_tools.conv_embeddings import tokenize_conv_task, CustomMistralModelConvEmbedding
from arc_tartiflette.graph.arc_grid import ArcGrid, get_default_arc_token_mapping

class ConvEmbeddingSolver(Solver):
    """
    Solver that uses a language model modified with my conv embedding class to solve the task
    """
    def __init__(
            self,
            model_name: str="",
            model_revision: str=None,
            model: CustomMistralModelConvEmbedding=None,
            tokenizer: AutoTokenizer=None,
            do_attempts: bool = True,
            sample_batch_size: int = 16,
        ):
        self.do_attempts = do_attempts
        self.model = model
        self.tokenizer = tokenizer
        if model == None or tokenizer == None:
            self.model_name = model_name
            self.model_revision = model_revision
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
            self.model = CustomMistralModelConvEmbedding.from_pretrained(model_name, revision=model_revision, device_map="auto", trust_remote_code=True)
        self.format = get_architects_prompt_format(self.tokenizer)
        self.token_mapping = get_default_arc_token_mapping(self.tokenizer)
        self.alternate_eos_token_id = self.tokenizer.eos_token_id
        self.sample_batch_size = sample_batch_size
        self.do_attempts = do_attempts

    def solve(self, task: dict, logs="") -> dict[str, np.ndarray]:
        return self.solve_batch([task], logs=logs)[0]
    

    def pad_task(self, tokenized_task, max_length, padding_side="left", pad_token_id=0):
        pad_size = max_length - tokenized_task["input_ids"].shape[0]
        pad_input_ids = torch.tensor([[pad_token_id]*8], dtype=torch.long).repeat(pad_size, 1)
        pad_position_ids = torch.tensor([[0,0]], dtype=torch.long).repeat(pad_size, 1)
        attention_mask = torch.tensor([1]*(self.sample_batch_size-pad_size) + [0]*pad_size, dtype=torch.long)
        if padding_side == "left":
            return {
                "input_ids": torch.cat([pad_input_ids, tokenized_task["input_ids"]], dim=0),
                "position_ids": torch.cat([pad_position_ids, tokenized_task["position_ids"]], dim=0),
            }, attention_mask
        else:
            return {
                "input_ids": torch.cat([tokenized_task["input_ids"], pad_input_ids], dim=0),
                "position_ids": torch.cat([tokenized_task["input_ids"], pad_position_ids], dim=0),
            }, attention_mask

    
    def select_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        Args:
            logits: [batch_size, num_candidates, vocab_size]
            temperature: float, softmax temperature

        Returns:
            selected_candidate_idx: [batch_size] LongTensor
            selected_token_id: [batch_size] LongTensor
        """
        # (1) Apply temperature
        logits = logits / temperature

        # (2) Compute probabilities per candidate
        probs = F.softmax(logits, dim=-1)  # [B, C, V]

        # (3) Sample one token ID for each candidate
        candidate_selected_token_id = torch.multinomial(
            probs.view(-1, probs.size(-1)), num_samples=1
        ).view(*probs.shape[:2])  # [B, C]

        # (4) Get its logit value
        candidate_selected_logit = torch.gather(logits, dim=-1, index=candidate_selected_token_id)

        # (5) Find the candidate with the highest selected logit
        selected_candidate_idx = candidate_selected_logit.argmax(dim=-1)

        # (6) Get the token id for the selected candidate
        selected_token_id = torch.gather(candidate_selected_token_id, dim=-1, index=selected_candidate_idx)

        return selected_candidate_idx, selected_token_id
    

    def append_selected_cache(self, old_cache, new_cache, candidate_idx):
        """
        Keep only the KV cache of the selected token for the rest of the generation
        """

        layer_shape = new_cache.layers[0].keys.shape
        index = candidate_idx.view(-1, 1, 1, 1).expand(-1, layer_shape[1], 1, layer_shape[3])

        old_seq_len = old_cache.layers[0].keys.shape[2]
        for i in range(len(new_cache.layers)):
            new_keys = new_cache.layers[i].keys[:, :, old_seq_len:, :]
            new_values = new_cache.layers[i].values[:, :, old_seq_len:, :]
            selected_key = torch.gather(new_keys, dim=2, index=index)
            selected_value = torch.gather(new_values, dim=2, index=index)

            old_cache.layers[i].keys   = torch.cat([old_cache.layers[i].keys, selected_key], dim=2)
            old_cache.layers[i].values = torch.cat([old_cache.layers[i].values, selected_value], dim=2)
        
        return old_cache
    

    def inverse_token_mapping(self, token_id: int) -> int:
        for k, v in self.token_mapping.items():
            if isinstance(v, list):
                if token_id in v:
                    return k
            if v == token_id:
                return k
        return None


    def solve_batch(self, tasks: list[dict], logs="") -> list[dict[str, np.ndarray]]:
        max_len = 0

        # Duplicate tasks to make 2 attempts each
        if self.do_attempts:
            tasks = tasks * 2
        
        batch_size = len(tasks)

        tokenized_tasks = []
        for task in tasks:
            tokenized_tasks.append(tokenize_conv_task(task, self.tokenizer, prompt=True))
            max_len = max(max_len, tokenized_tasks[-1]["input_ids"].shape[0])

        # Pad to max length
        input_ids_list = []
        position_ids_list = []
        attention_mask_list = []
        for task in tokenized_tasks:
            task, attention_mask = self.pad_task(task, max_len, padding_side="left", pad_token_id=self.tokenizer.pad_token_id)
            input_ids_list.append(task["input_ids"].unsqueeze(0))
            position_ids_list.append(task["position_ids"].unsqueeze(0))
            attention_mask_list.append(attention_mask.unsqueeze(0))
        
        # Collate batches
        input_ids = torch.cat(input_ids_list, dim=0).to(self.model.device)
        position_ids = torch.cat(position_ids_list, dim=0).to(self.model.device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(self.model.device)

        # Explore grids
        arc_grids = [ArcGrid.for_generation(self.token_mapping, name=f"grid_{i}") for i in range(batch_size)]
        print("Initial grids:")
        for i, g in enumerate(arc_grids):
            print(f"Grid {i}:")
            print(g)
        for i, g in enumerate(arc_grids):
            g.assign_value(g.nodes[0][0], -1)

        # Prompt inference
        logits, cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )[:2]

        explorable_nodes = [g.get_explorable_nodes() for g in arc_grids]
        for nodes in explorable_nodes:
            random.shuffle(nodes)
        print("Explorable nodes:", explorable_nodes)
        while not all(exp_nodes == [] for exp_nodes in explorable_nodes):

            # Tokenize and pad potential next nodes candidates
            for nodes in explorable_nodes:
                tokenized_nodes = [n.tokenized(token_mapping=self.token_mapping) for n in nodes[:self.sample_batch_size]]
                print(f"Tokenized nodes: {tokenized_nodes}")

                input_ids_pad = [self.tokenizer.pad_token_id]*8
                position_ids_pad = [0, 0]
                pad_length = self.sample_batch_size - len(tokenized_nodes)

                task_input_ids = [tok_n["input_ids"] for tok_n in tokenized_nodes] + [input_ids_pad]*pad_length
                task_position_ids = [tok_n["position_ids"] for tok_n in tokenized_nodes] + [position_ids_pad]*pad_length

                input_ids_list.append([task_input_ids])
                position_ids_list.append([task_position_ids])
                attention_mask_list.append([[1]*len(tokenized_nodes) + [0]*pad_length])

            # Collate batches
            input_ids = torch.cat(input_ids_list, dim=0).to(self.model.device)
            position_ids = torch.cat(position_ids_list, dim=0).to(self.model.device)
            attention_mask = torch.cat(attention_mask_list, dim=0).to(self.model.device)
            print(f"Input IDs shape: {input_ids.shape}, Position IDs shape: {position_ids.shape}, Attention mask shape: {attention_mask.shape}")

            # Infer model
            logits, new_cache = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                attention_mask=attention_mask,
            )[:2]

            # Select which candidate to pick
            candidate_idx, token_id = self.select_next_token(logits, self.temperature)

            # Keep cache only for that candidate
            cache = self.append_selected_cache(old_cache=cache, new_cache=new_cache, candidate_idx=candidate_idx)

            # Update grids
            for i, g in enumerate(arc_grids):
                if torch.sum(attention_mask[i]) == 0:
                    assert explorable_nodes[i] == []
                    continue
                try:
                    color_id = self.inverse_token_mapping(token_id[i].item())
                except:
                    token_str = self.tokenizer.decode(token_id[i].item()).strip()
                    print(f"Invalid token decoded: {token_str}, assigning color 0")
                    color_id = 0
                node = explorable_nodes[i][candidate_idx[i].item()]
                g.assign_value(node, color_id)
            
            # Print grids
            print("Current grids state:")
            for i, g in enumerate(arc_grids):
                print(f"Grid {i}:")
                print(g)

            # Get reacheable nodes for next inference
            explorable_nodes = [g.get_explorable_nodes() for g in arc_grids]
            for nodes in explorable_nodes:
                random.shuffle(nodes)
            for i, nodes in enumerate(explorable_nodes):
                print(f"Explorable nodes for grid {i}: {[n.grid_position for n in nodes]}")

        # Extract solutions
        attempts_list = []
        if self.do_attempts:
            for i in range(batch_size//2):
                attempts_list.append({
                    "attempt_1": arc_grids[i].extract_2D_grid(),
                    "attempt_2": arc_grids[i + (batch_size//2)].extract_2D_grid(),
                })
        else:
            for i in range(batch_size):
                attempts_list.append({
                    "attempt_1": arc_grids[i].extract_2D_grid(),
                    "attempt_2": arc_grids[i].extract_2D_grid(),
                })

        return attempts_list
