import logging

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.mistral.modeling_mistral import (
    MistralRotaryEmbedding,
    MistralModel,
    MistralForCausalLM,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from datasets import DatasetDict, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_tartiflette.model.custom_pe import (
    CustomRotaryEmbedding2D,
    CustomCompletionMaskDataCollator,
)
from arc_tartiflette.model.tokenizer_tools import get_architects_prompt_format
from arc_tartiflette.dataset.tokenize_functions import make_completion_mask
from arc_tartiflette.graph.arc_grid import get_default_arc_token_mapping, ArcGrid
from arc_tartiflette.dataset.tokenizer_utils import pad_and_truncate, pad_and_truncate_batch

logger = logging.getLogger(__name__)


class CustomConvEmbedding(nn.Embedding):
    """
    Custom embedding that expects input of shape (batch_size, seq_len, num_subtokens),
    sums all embeddings of the subtokens (ignoring ID=0), and normalizes each summed vector.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_subtokens: int = 8,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.num_subtokens = num_subtokens
        self.norm = nn.RMSNorm(embedding_dim, eps=eps)  # Same as MistralRMSNorm
        self.oob_token_id = 0  # TODO: Change to tokenizer oob token

    def forward(self, input: torch.Tensor) -> torch.Tensor: # pylint: disable=redefined-builtin
        """
        input: LongTensor of shape (batch_size, seq_len, num_subtokens)
        output: Tensor of shape (batch_size, seq_len, embedding_dim)
        """

        # (B, L, S, D)
        embeds = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Mask out special tokens (ID == 0)
        mask = (input != self.oob_token_id).unsqueeze(-1)  # (B, L, S, 1)
        embeds = embeds * mask  # zero-out embeddings of special tokens

        # Sum over subtoken dimension (S)
        summed = embeds.sum(dim=2)  # (B, L, D)

        # Normalize each token embedding vector
        normed = self.norm(summed)

        return normed


class CustomMistralModelConvBase(MistralModel): # pylint: disable=abstract-method
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


class CustomMistralModelConvEmbedding(MistralForCausalLM): # pylint: disable=abstract-method
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModelConvBase(config)
        self.post_init()
        logger.info("CustomMistralModelConvEmbedding initialized, post_init called.")
        logger.debug("CustomMistralModelConvEmbedding.config: %s", self.config)
        logger.debug(
            "CustomMistralModelConvEmbedding.generation_config: %s",
            self.generation_config,
        )


def tokenize_simple_char(
    char: str = None,
    tokenizer: PreTrainedTokenizerFast = None,
    current_position: list[int] | None = None,
    tok_id: int = None,
) -> dict:
    if current_position is None:
        current_position = [0, 0]
    token_mapping = get_default_arc_token_mapping(tokenizer)
    char_token_id = tok_id if tok_id else tokenizer(str(char))["input_ids"][-1]
    tokenized = {
        "input_ids": torch.tensor(
            [[char_token_id] + [token_mapping["out_of_bounds"]] * 7], dtype=torch.long
        ),
        "position_ids": torch.tensor([list(current_position)], dtype=torch.long),
        "labels": torch.tensor([char_token_id], dtype=torch.long),
    }
    return tokenized


def tokenize_conv_grid(
    grid: list[list[int]],
    tokenizer: PreTrainedTokenizerFast,
    current_position: tuple[int, int] | None = None,
):
    if current_position is None:
        current_position = (0, 0)
    # Create token mapping
    token_mapping = get_default_arc_token_mapping(tokenizer)

    # Create ArcGrid
    arc_grid = ArcGrid(grid, token_mapping)

    # Perform random exploration
    tokenized_path = arc_grid.random_exploration()

    # add current position to position_ids (shape is [seq_len, 2])
    tokenized_path["position_ids"] = tokenized_path["position_ids"] + np.array(
        current_position
    )

    grid_end = (
        current_position[0] + arc_grid.height,
        current_position[1] + arc_grid.width,
    )

    return tokenized_path, grid_end


def tokenize_conv_example(
    example: dict,
    tokenizer: PreTrainedTokenizerFast,
    current_position: list[int] = (0, 0),
    mask_output=False,
):
    fmt = get_architects_prompt_format(tokenizer)

    tokenized = []

    for grid in ["input", "output"]:
        if grid == "input":
            tokenized.append(
                tokenize_simple_char(
                    tok_id=fmt["input_beg_id"],
                    tokenizer=tokenizer,
                    current_position=current_position,
                )
            )
        else:
            tokenized.append(
                tokenize_simple_char(
                    tok_id=fmt["output_beg_id"],
                    tokenizer=tokenizer,
                    current_position=current_position,
                )
            )

        current_position = (current_position[0] + 1, current_position[1] + 1)
        if not mask_output or grid == "input":
            tokenized_grid, new_pos = tokenize_conv_grid(
                example[grid],
                tokenizer,
                current_position=current_position,
            )
            tokenized.append(tokenized_grid)
            current_position = new_pos

    return {
        "input_ids": torch.cat([t["input_ids"] for t in tokenized], dim=0),
        "position_ids": torch.cat([t["position_ids"] for t in tokenized], dim=0),
        "labels": torch.cat([t["labels"] for t in tokenized], dim=0),
    }, current_position


def tokenize_conv_task(task: dict, tokenizer: PreTrainedTokenizerFast, prompt=False):
    fmt = get_architects_prompt_format(tokenizer)

    tokenized = []
    current_position = [0, 0]

    # Task beg
    tokenized.append(
        tokenize_simple_char(
            tok_id=fmt["bos_token_id"], tokenizer=tokenizer, current_position=(0, 0)
        )
    )
    current_position = [current_position[0] + 1, current_position[1] + 1]
    for char in fmt["preprompt"]:
        tokenized.append(
            tokenize_simple_char(
                char=char, tokenizer=tokenizer, current_position=current_position
            )
        )
        current_position = (current_position[0] + 1, current_position[1] + 1)

    for i, example in enumerate(task["train"] + task["test"]):
        if i > 0:
            # Example separator
            tokenized.append(
                tokenize_simple_char(
                    tok_id=fmt["bos_token_id"],
                    tokenizer=tokenizer,
                    current_position=current_position,
                )
            )
            current_position = (current_position[0] + 1, current_position[1] + 1)

        mask_output = prompt and i == (
            len(task["train"] + task["test"]) - 1
        )  # Mask last output

        tokenized_example, new_pos = tokenize_conv_example(
            example, tokenizer, current_position, mask_output=mask_output
        )
        tokenized.append(tokenized_example)
        current_position = new_pos

        if not mask_output:
            tokenized.append(
                tokenize_simple_char(
                    tok_id=fmt["eos_token_id"],
                    tokenizer=tokenizer,
                    current_position=current_position,
                )
            )
            current_position = (current_position[0] + 1, current_position[1] + 1)

    return {
        "input_ids": torch.cat([t["input_ids"] for t in tokenized], dim=0),
        "position_ids": torch.cat([t["position_ids"] for t in tokenized], dim=0),
        "labels": torch.cat([t["labels"] for t in tokenized], dim=0),
    }


def tokenize_row_conv(
    ds_row,
    tokenizer: AutoTokenizer,
    max_length,
    padding="max_length",
    truncation=True,
):
    """
    Tokenizes a batch of tasks with 2D positional encoding.
    """

    task = ds_row["task"]

    tokenized = tokenize_conv_task(task, tokenizer)

    logger.debug("Tokenized row before padding/truncation: %s", tokenized)

    # Pad and truncate input_ids, position_ids, labels
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_input_id = torch.full((8,), pad_token_id, dtype=torch.long)
    pad_position_id = torch.full((2,), 0, dtype=torch.long)

    ds_row["input_ids"] = pad_and_truncate(
        tokenized["input_ids"], 
        max_length=max_length, 
        pad_value=pad_input_id,
    )
    ds_row["position_ids"] = pad_and_truncate(
        tokenized["position_ids"],
        max_length=max_length,
        pad_value=pad_position_id,
    )
    ds_row["labels"] = pad_and_truncate(
        tokenized["labels"],
        max_length=max_length, 
        pad_value=-100,
    )
    ds_row["attention_mask"] = torch.where(ds_row["labels"] != -100, 1, 0)
    return ds_row


def tokenize_dataset_conv(dataset_dict: DatasetDict, tokenizer: AutoTokenizer, max_length=None):
    logger.info("Tokenizing dataset with Conv tokenizer...")
    if max_length is None:
        max_length = tokenizer.model_max_length
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

    tokenized_datasets = dataset_dict.map(
        tokenize_row_conv,
        fn_kwargs={
            "max_length": max_length,
            "tokenizer": tokenizer,
            "padding": padding,
            "truncation": truncation,
        },
        batched=False,
    )
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
    logger.info("Dataset tokenized with Conv tokenizer.")
    logger.debug(
        "Tokenized dataset example: %s",
        (
            tokenized_datasets["train"][0]
            if len(tokenized_datasets["train"]) > 0
            else "N/A"
        ),
    )
    return tokenized_datasets


if __name__ == "__main__":  # Test the conv tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    example_task = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
        "test": [{"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]}],
    }
    tokenizer.pad_token = (
        tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
    )
    row = {"task": example_task}
    tokenized = tokenize_row_conv(row, tokenizer, max_length=128)
    print("Tokenized example input_ids:", tokenized["input_ids"])
    print("Tokenized example position_ids:", tokenized["position_ids"])
    print("Tokenized example labels:", tokenized["labels"])
    print("Tokenized example attention_mask:", tokenized["attention_mask"])

    hf_dataset = Dataset.from_list([row])
    dataset_dict = DatasetDict(
        {
            "train": hf_dataset,
        }
    )
    tokenized_datasets = tokenize_dataset_conv(dataset_dict, tokenizer)
    print("Tokenized dataset example input_ids:", tokenized_datasets["train"][0])
