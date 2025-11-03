from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralModel, MistralForCausalLM
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from datasets import DatasetDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_tartiflette.model_tools.custom_pe import CustomRotaryEmbedding2D, CustomCompletionMaskDataCollator
from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format
from arc_tartiflette.model_tools.tokenize_functions import make_completion_mask
from arc_tartiflette.graph.arc_grid import get_default_arc_token_mapping, ArcGrid
from arc_tartiflette.config.settings import ENV_VARS


class CustomConvEmbedding(nn.Embedding):
    """
    Custom embedding that expects input of shape (batch_size, seq_len, num_subtokens),
    sums all embeddings of the subtokens (ignoring ID=0), and normalizes each summed vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, num_subtokens: int = 8, eps: float = 1e-6, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.num_subtokens = num_subtokens
        self.norm = nn.RMSNorm(embedding_dim, eps=eps)  # Same as MistralRMSNorm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: LongTensor of shape (batch_size, seq_len, num_subtokens)
        output: Tensor of shape (batch_size, seq_len, embedding_dim)
        """

        # (B, L, S, D)
        embeds = F.embedding(input, self.weight, self.padding_idx, 
                             self.max_norm, self.norm_type, 
                             self.scale_grad_by_freq, self.sparse)

        # Mask out special tokens (ID == 0)
        mask = (input != 0).unsqueeze(-1)  # (B, L, S, 1)
        embeds = embeds * mask  # zero-out embeddings of special tokens

        # Sum over subtoken dimension (S)
        summed = embeds.sum(dim=2)  # (B, L, D)

        # Normalize each token embedding vector
        normed = self.norm(summed)

        return normed


class CustomMistralModelConvBase(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = CustomRotaryEmbedding2D(config)
        self.embed_tokens = CustomConvEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            num_subtokens=8,  # Assuming 8 subtokens per token
            eps=config.rms_norm_eps,
            padding_idx=self.padding_idx,
        )


class CustomMistralModelConvEmbedding(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModelConvBase(config)
        self.post_init()
        print("After post_init:")
        print("CustomMistralModelConvEmbedding.config:", self.config)
        print("CustomMistralModelConvEmbedding.generation_config:", self.generation_config)
        print("CustomMistralModelConvEmbedding.model.config:", self.model.config)
        print("CustomMistralModelConvEmbedding.model.generation_config:", self.model.generation_config)


def tokenize_simple_char(char: str, tokenizer: PreTrainedTokenizerFast, current_position: tuple[int, int], id:int=None) -> list[int]:
    token_mapping = get_default_arc_token_mapping(tokenizer)
    char_token_id = id if id else tokenizer(char)["input_ids"][-1]
    tokenized = {
        "input_ids": torch.tensor([[char_token_id] + [token_mapping["out_of_bounds"]]*7], dtype=torch.long),
        "position_ids": torch.tensor([list(current_position)], dtype=torch.long),
        "labels": torch.tensor([char_token_id], dtype=torch.long)
    }
    return tokenized,


def tokenize_conv_grid(grid: list[list[int]], tokenizer: PreTrainedTokenizerFast, current_position: tuple[int, int]=(0,0)):
    # Create token mapping
    token_mapping = get_default_arc_token_mapping(tokenizer)

    # Create ArcGrid
    arc_grid = ArcGrid(grid, token_mapping)

    # Perform random exploration
    tokenized_path = arc_grid.random_exploration()

    # add current position to position_ids (shape is [seq_len, 2])
    tokenized_path["position_ids"] = tokenized_path["position_ids"] + np.array(current_position)

    grid_end = current_position[0] + arc_grid.height, current_position[1] + arc_grid.width

    return tokenized_path, grid_end


def tokenize_conv_example(example: dict, tokenizer: PreTrainedTokenizerFast, current_position: list[int]=(0,0)):
    fmt = get_architects_prompt_format(tokenizer)

    tokenized = []

    for grid in ["input", "output"]:
        if grid == "input":
            tokenized.append(tokenize_simple_char(id=fmt["input_beg_id"]), tokenizer=tokenizer, current_position=current_position)
        else:
            tokenized.append(tokenize_simple_char(id=fmt["output_beg_id"]), tokenizer=tokenizer, current_position=current_position)
        current_position = (current_position[0]+1, current_position[1]+1)
        tokenized_grid, new_pos = tokenize_conv_grid(
            example[grid],
            tokenizer,
            current_position=current_position,
        )
        tokenized.append(tokenized_grid)
        current_position = new_pos
    return {
        "input_ids": torch.cat([t["input_ids"] for t in tokenized], dim=1),
        "position_ids": torch.cat([t["position_ids"] for t in tokenized], dim=1),
        "labels": torch.cat([t["labels"] for t in tokenized], dim=1),
    }, current_position

def tokenize_conv_task(task: dict, tokenizer: PreTrainedTokenizerFast):
    fmt = get_architects_prompt_format(tokenizer)

    tokenized = []

    # Task beg
    tokenized.append(tokenize_simple_char(id=fmt["bos_token_id"], tokenizer=tokenizer, current_position=(0,0)))
    current_position = [current_position[0]+1, current_position[1]+1]
    for char in fmt["preprompt"]:
        tokenized.append(tokenize_simple_char(char=char, tokenizer=tokenizer, current_position=current_position))
        current_position = (current_position[0]+1, current_position[1]+1)


    for i, example in enumerate(task["train"] + task["test"]):
        if i > 0:
            # Example separator
            tokenized.append(tokenize_simple_char(id=fmt["bos_token_id"], tokenizer=tokenizer, current_position=current_position))
            current_position = (current_position[0]+1, current_position[1]+1)

        tokenized_example, new_pos = tokenize_conv_example(example, tokenizer, current_position)
        tokenized.append(tokenized_example)
        current_position = new_pos

        tokenized.append(tokenize_simple_char(id=fmt["eos_token_id"], tokenizer=tokenizer, current_position=current_position))
        current_position = (current_position[0]+1, current_position[1]+1)


    return {
        "input_ids": torch.cat([t["input_ids"] for t in tokenized], dim=1),
        "position_ids": torch.cat([t["position_ids"] for t in tokenized], dim=1),
        "labels": torch.cat([t["labels"] for t in tokenized], dim=1),
    }

def tokenize_row_conv(ds_row, tokenizer: AutoTokenizer, max_length=4096, padding="max_length", truncation=True):
    """
    Tokenizes a batch of tasks with 2D positional encoding.
    """

    task = ds_row["task"]

    tokenized = tokenize_conv_task(task, tokenizer)

    if truncation and len(tokenized["input_ids"]) > max_length:
        tokenized["input_ids"] = tokenized["input_ids"][:max_length]
        tokenized["position_ids"] = tokenized["position_ids"][:max_length]
        tokenized["labels"] = tokenized["labels"][:max_length]
    elif padding == "max_length" and len(tokenized["input_ids"]) < max_length:
        pad_length = max_length - len(tokenized["input_ids"])
        tokenized["input_ids"].extend([[tokenizer.pad_token_id]*8] * pad_length)
        tokenized["position_ids"].extend([[0,0]] * pad_length)
        tokenized["labels"].extend([-100] * pad_length)

    attention_mask = [1 if id != -100 else 0 for id in tokenized["labels"]]

    ds_row["input_ids"] = torch.tensor(tokenized["input_ids"]).long()
    ds_row["position_ids"] = torch.tensor(tokenized["position_ids"]).long()
    ds_row["attention_mask"] = torch.tensor(attention_mask).long()
    ds_row["labels"] = torch.tensor(tokenized["labels"]).long()
    return ds_row


def tokenize_dataset_conv(dataset_dict: DatasetDict, tokenizer: AutoTokenizer):
    print("Tokenizing dataset with Conv tokenizer...")
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]
    padding = "max_length"
    truncation = True

    fmt = get_architects_prompt_format(tokenizer)

    def tokenize_function(example):
        tokenized = example
        tokenized["completion_mask"] = make_completion_mask(
            tokenized["labels"], 
            tokenized["attention_mask"], 
            special_token_id=tokenizer("I")["input_ids"][0],
            n=1,
        )
        return tokenized
    tokenized_datasets = dataset_dict.map(tokenize_row_conv, fn_kwargs={"max_length": max_length, "tokenizer": tokenizer, "padding": padding, "truncation": truncation}, batched=False)
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
    print("---- Dataset tokenized with Conv tokenizer. ----")
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")
    return tokenized_datasets
