from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
import numpy as np
import torch


class CustomRotaryEmbedding2D(MistralRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        x_inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        y_inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        x_position_ids = position_ids[..., 0]
        y_position_ids = position_ids[..., 1]
        x_position_ids_expanded = x_position_ids[:, None, :].float()
        y_position_ids_expanded = y_position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            x_freqs = (x_inv_freq_expanded.float() @ x_position_ids_expanded.float())
            y_freqs = (y_inv_freq_expanded.float() @ y_position_ids_expanded.float())
            freqs = (x_freqs + y_freqs).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class TokenizerWrapper2D:
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
        self.vocab = base_tokenizer.get_vocab()
        self.pad_token_id = base_tokenizer.pad_token_id or 0

    def encode_grid(self, grid: np.ndarray):
        """
        Your custom logic:
        - map each grid element to a token ID
        - reuse existing tokenizer IDs where possible
        """
        # Example: flatten the grid and map numbers to vocab IDs
        input_ids = [self.vocab.get(str(int(x)), self.tokenizer.unk_token_id) for x in grid.flatten()]
        return input_ids

    def __call__(self, data_dict):
        """
        data_dict: {"grid": np.array, "text": str, "label": int, ...}
        """
        input_ids = self.encode_grid(data_dict["grid"])
        
        # Optionally include text tokens
        if "text" in data_dict:
            text_ids = self.tokenizer.encode(data_dict["text"], add_special_tokens=False)
            input_ids.extend(text_ids)
        
        attention_mask = [1] * len(input_ids)
        
        # Convert to tensors and pad to max length
        max_len = 128  # or dynamic
        input_ids = input_ids[:max_len] + [self.pad_token_id] * (max_len - len(input_ids))
        attention_mask = attention_mask[:max_len] + [0] * (max_len - len(attention_mask))
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(data_dict.get("label", -100))  # -100 for ignore
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