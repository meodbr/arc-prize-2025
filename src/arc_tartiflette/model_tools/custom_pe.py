from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralModel, MistralForCausalLM
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
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


class CustomCompletionMaskDataCollator(DataCollatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_tensors = "pt"

    def torch_call(self, examples, **kwargs):
        batch = {}
        print(examples)
        batch["input_ids"] = torch.stack([torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples])
        batch["position_ids"] = torch.stack([torch.tensor(ex["position_ids"], dtype=torch.long) for ex in examples])
        batch["attention_mask"] = torch.stack([torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples])
        batch["labels"] = torch.stack([torch.tensor(ex["labels"], dtype=torch.long) for ex in examples])
        # For every 0 in completion_mask, set corresponding label to -100
        batch["labels"] = batch["labels"].masked_fill(
            torch.stack([torch.tensor(ex["completion_mask"], dtype=torch.long) for ex in examples]) == 0,
            -100
        )

        return batch

class CustomMistralModel2DPEBase(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = CustomRotaryEmbedding2D(config)


class CustomMistralModel2DPE(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModel2DPEBase(config)
        self.post_init()
        print("After post_init:")
        print("CustomMistralModel2DPE.config:", self.config)
        print("CustomMistralModel2DPE.generation_config:", self.generation_config)
        print("CustomMistralModel2DPE.model.config:", self.model.config)
        print("CustomMistralModel2DPE.model.generation_config:", self.model.generation_config)


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