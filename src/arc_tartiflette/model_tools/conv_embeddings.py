from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralModel, MistralForCausalLM
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_tartiflette.model_tools.custom_pe import CustomRotaryEmbedding2D, CustomCompletionMaskDataCollator
from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format
from arc_tartiflette.graph.arc_grid import get_default_arc_token_mapping, ArcGrid


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


def tokenize_conv_example(example: dict, tokenizer: PreTrainedTokenizerFast, current_position: tuple[int, int]=(0,0)):
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
    }

def tokenize_conv_task(task: dict, tokenizer: PreTrainedTokenizerFast):
    fmt = get_architects_prompt_format(tokenizer)

    tokenized = []

    # Task beg
    tokenized.append(tokenize_simple_char(id=fmt["bos_token_id"], tokenizer=tokenizer, current_position=(0,0)))
    for char in fmt["preprompt"]:
        tokenized.append(tokenize_simple_char(char=char, tokenizer=tokenizer, current_position=(0,0)))

    current_position = (1,1)

    for example in task["examples"]:
        tokenized_example = tokenize_conv_example(example, tokenizer, current_position)
        tokenized.append(tokenized_example)
        current_position = (current_position[0]+len(example["input"])+2, current_position[1]+len(example["input"][0])+2)

    # Task end
    tokenized.append(tokenize_simple_char(id=fmt["task_end_id"], tokenizer=tokenizer, current_position=current_position))

    return {
        "input_ids": torch.cat([t["input_ids"] for t in tokenized], dim=1),
        "position_ids": torch.cat([t["position_ids"] for t in tokenized], dim=1),
        "labels": torch.cat([t["labels"] for t in tokenized], dim=1),
    }


# BASE CLASS (kept here to have an example)

# class MistralRotaryEmbedding(nn.Module):
#     def __init__(self, config: MistralConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
#         position_ids_expanded = position_ids[:, None, :].float()

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)