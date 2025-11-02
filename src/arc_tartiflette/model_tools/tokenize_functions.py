import torch
from transformers import AutoTokenizer
from datasets import DatasetDict
import numpy as np

from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format


def make_completion_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_token_id: int, n: int) -> torch.Tensor:
    """
    Creates a mask that zeros out all tokens up to and including the nth occurrence
    of a specific token per sequence.
    
    Args:
        input_ids (torch.Tensor): [batch, seq_len]
        special_token_id (int): token ID to count
        n (int): number of occurrences to skip before computing loss
    
    Returns:
        torch.Tensor: [batch, seq_len] mask (1 = keep, 0 = ignore)
    """
    # Boolean mask for where the special token occurs
    input_ids = torch.tensor(input_ids)
    special_mask = (input_ids == special_token_id).int()

    # Compute cumulative count of special tokens along sequence dimension
    cumsum = torch.cumsum(special_mask, dim=1)

    # Create the mask: 1 where we are past the nth occurrence, else 0
    loss_mask = (cumsum >= n).int()

    # Zero out positions before the nth occurrence
    # Optionally exclude the nth token itself:
    loss_mask = torch.where(cumsum > n, 1, 0)

    if attention_mask != None and isinstance(attention_mask, torch.Tensor) and attention_mask.shape == loss_mask.shape:
        loss_mask = loss_mask & attention_mask

    return loss_mask


def replace_bos_eos_batch(batch, tokenizer, text_column="text"):
    """
    Replaces literal <s> and </s> in a batch of texts with the tokenizer's BOS and EOS tokens.

    Args:
        batch (dict): Batch from datasets.map with key text_column
        tokenizer: Hugging Face tokenizer instance
        text_column (str): Name of the column containing text

    Returns:
        dict: Batch with updated text
    """
    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"

    # Replace in all texts in the batch
    batch[text_column] = [
        t.replace("<s>", bos).replace("</s>", eos) for t in batch[text_column]
    ]
    return batch


def tokenize_dataset_base(dataset_dict: DatasetDict, tokenizer: AutoTokenizer):
    print("Tokenizing dataset...")
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]

    def tokenize_function(example):
        if tokenizer.bos_token != "<s>":
            example = replace_bos_eos_batch(example, tokenizer)
        tokenized = tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["completion_mask"] = make_completion_mask(
            tokenized["input_ids"], 
            tokenized["attention_mask"], 
            special_token_id=tokenizer("I")["input_ids"][0],
            n=1,
        )
        return tokenized
    print(f"Example before tokenization:", dataset_dict['train'][0] if len(dataset_dict['train']) > 0 else "N/A")
    if "text" not in dataset_dict['train'].column_names:
        raise ValueError(f"Dataset does not contain 'text' column. Available columns: {dataset_dict['train'].column_names}")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print("---- Dataset tokenized. ----")
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")
    return tokenized_datasets


def compute_2DPE_pos_ids(example, tokenizer: AutoTokenizer):
    """
    Computes 2D positional IDs for a given example.
    Not batched
    """
    format = get_architects_prompt_format(tokenizer)
    input_ids = example["input_ids"]
    x, y = 0, 0
    pos_ids = []
    grid_origin = [0, 0]
    input_beg_id = tokenizer(format["input_beg"])["input_ids"][0]
    output_beg_id = tokenizer(format["output_beg"])["input_ids"][0]
    row_end_id = tokenizer(format["row_end"])["input_ids"][0]
    is_in_grid = False
    for i in range(len(input_ids)):
        pos_ids.append([x, y])
        if input_ids[i] == tokenizer.eos_token_id:
            is_in_grid = False
        if input_ids[i] in [input_beg_id, output_beg_id]:
            grid_origin = [x, y]
            is_in_grid = True

        # How x and y are assigned depends on whether we are in the grid or not
        if input_ids[i] == row_end_id and is_in_grid:
            y += 1
            x = grid_origin[0]
            print(tokenizer.decode(input_ids[i]), end="\n")
        elif is_in_grid:
            print(tokenizer.decode(input_ids[i]), end="")
            x += 1
        else:
            print(tokenizer.decode(input_ids[i]), end="-")
            x += 1
            y += 1
    example["position_ids"] = torch.Tensor(pos_ids).long()

    return example


def get_token_mapping(tokenizer: AutoTokenizer):
    """
    Returns a mapping from characters to token IDs for the tokenizer.
    Assumes single-character tokens are used for grid elements.
    """
    vocab = tokenizer.get_vocab()
    token_mapping = []
    for i in range(10): # digits 0-9
        char = str(i)
        if char in vocab:
            token_mapping.append(vocab[char])
    return token_mapping


def tokenize_2DPE_grid(grid, tokenizer: AutoTokenizer, grid_origin=(0,0), digit_mapping=None):
    """
    Computes input IDs and 2D positional IDs for the grid part of a given example.
    Not batched
    """
    x, y = grid_origin
    fmt = get_architects_prompt_format(tokenizer)
    row_end_id = fmt["row_end_id"]
    input_ids = []
    pos_ids = []
    for i, row in enumerate(grid):
        for num in row:
            input_ids.append(digit_mapping[num])  # Map char to token ID using tokenizer
            pos_ids.append([x, y])
            x += 1
        input_ids.append(row_end_id)
        pos_ids.append([x, y])
        y += 1
        if i < len(grid) - 1:
            x = grid_origin[0]
    grid_end = (x+1, y+1)
    return input_ids, pos_ids, grid_end


def tokenize_example_2DPE(example, tokenizer: AutoTokenizer, digit_mapping=None, input_beg_coord=(0,0)):
    """
    Tokenizes a single example with 2D positional encoding.
    Not batched
    """
    fmt = get_architects_prompt_format(tokenizer)

    input_ids = []
    position_ids = []
    x, y = input_beg_coord
    input_ids.append(fmt["input_beg_id"])
    position_ids.append([x, y])
    x += 1
    y += 1

    input_input_ids, input_pos_ids, (x, y) = tokenize_2DPE_grid(
        example["input"], tokenizer, grid_origin=(x,y), digit_mapping=digit_mapping
    )
    input_ids.extend(input_input_ids)
    position_ids.extend(input_pos_ids)
    input_ids.append(fmt["output_beg_id"])
    position_ids.append([x, y])
    x += 1
    y += 1

    output_input_ids, output_pos_ids, (x, y) = tokenize_2DPE_grid(
        example["output"], tokenizer, grid_origin=(x,y), digit_mapping=digit_mapping
    )
    input_ids.extend(output_input_ids)
    position_ids.extend(output_pos_ids)

    return input_ids, position_ids, (x, y)

def tokenize_task_2DPE(task, tokenizer: AutoTokenizer, digit_mapping=None):
    """
    Tokenizes a full task (multiple examples) with 2D positional encoding.
    Not batched
    """
    fmt = get_architects_prompt_format(tokenizer)

    input_ids = []
    position_ids = []
    x, y = 0, 0

    preprompt_ids = [fmt["bos_token_id"]] + tokenizer(fmt["preprompt"])["input_ids"]
    preprompt_pos_ids = np.arange(len(preprompt_ids)).reshape(-1, 1).repeat(2, axis=1).tolist()
    input_ids.extend(preprompt_ids)
    position_ids.extend(preprompt_pos_ids)
    x += len(preprompt_ids)
    y += len(preprompt_ids)

    for example in task["train"] + task["test"]:
        ex_input_ids, ex_pos_ids, (x, y) = tokenize_example_2DPE(
            example, tokenizer, digit_mapping=digit_mapping, input_beg_coord=(x,y)
        )
        input_ids.extend(ex_input_ids)
        position_ids.extend(ex_pos_ids)
        # Add example end token
        input_ids.append(tokenizer.eos_token_id)
        position_ids.append([x, y])
        x += 1
        y += 1

    return input_ids, position_ids


def get_grid_from_pos_ids(input_ids, pos_ids):
    """
    Reconstructs a 2D grid from the input IDs and their corresponding position IDs.
    """
    max_grid_shape = [34, 34]  # Assuming max grid size
    grid = np.zeros(max_grid_shape, dtype=int)
    for (input_id, (x, y)) in zip(input_ids, pos_ids):
        if x < max_grid_shape[0] and y < max_grid_shape[1]:
            grid[y][x] = input_id
    return grid


def print_grid_from_pos_ids(input_ids, pos_ids, tokenizer):
    """
    Prints a 2D grid from the input IDs and their corresponding position IDs.
    """
    current_y = 0
    grid = get_grid_from_pos_ids(input_ids, pos_ids)
    for y in range(len(grid)):
        row_str = ""
        for x in range(len(grid[y])):
            token_id = grid[y][x]
            token_str = tokenizer.decode([token_id])
            row_str += token_str
        print(row_str)


def tokenize_row_2DPE(ds_row, tokenizer: AutoTokenizer, max_length=4096, padding="max_length", truncation=True):
    """
    Tokenizes a batch of tasks with 2D positional encoding.
    """
    digit_mapping = get_token_mapping(tokenizer)


    task = ds_row["task"]

    input_ids, position_ids = tokenize_task_2DPE(task, tokenizer, digit_mapping=digit_mapping)

    if truncation and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        position_ids = position_ids[:max_length]
    elif padding == "max_length" and len(input_ids) < max_length:
        pad_length = max_length - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * pad_length)
        position_ids.extend([[0,0]] * pad_length)
    
    attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]

    ds_row["input_ids"] = torch.tensor(input_ids).long()
    ds_row["position_ids"] = torch.tensor(position_ids).long()
    ds_row["attention_mask"] = torch.tensor(attention_mask).long()
    return ds_row


def tokenize_dataset_2DPE(dataset_dict: DatasetDict, tokenizer: AutoTokenizer):
    print("Tokenizing dataset with 2DPE tokenizer...")
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]
    padding = "max_length"
    truncation = True

    format = get_architects_prompt_format(tokenizer)

    def tokenize_function(example):
        tokenized = example
        tokenized["input_ids"] = torch.tensor(tokenized["input_ids"]).long()
        tokenized["position_ids"] = torch.tensor(tokenized["position_ids"]).long()
        tokenized["attention_mask"] = torch.tensor(tokenized["attention_mask"]).long()

        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["completion_mask"] = make_completion_mask(
            tokenized["input_ids"], 
            tokenized["attention_mask"], 
            special_token_id=tokenizer("I")["input_ids"][0],
            n=1,
        )
        return tokenized
    tokenized_datasets = dataset_dict.map(tokenize_row_2DPE, fn_kwargs={"max_length": max_length, "tokenizer": tokenizer, "padding": padding, "truncation": truncation}, batched=False)
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
    print("---- Dataset tokenized with 2DPE tokenizer. ----")
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")
    return tokenized_datasets


def frac_dataset_dict(dataset_dict: DatasetDict, frac=0.1, seed=42):
    """
    Returns a fraction of each split in a DatasetDict.
    
    Args:
        dataset_dict (DatasetDict): original dataset
        frac (float): fraction to keep (0 < frac <= 1)
        seed (int): random seed for reproducibility
    
    Returns:
        DatasetDict: fractioned dataset
    """
    small_splits = {}
    for split_name, ds in dataset_dict.items():
        # train_test_split returns a dict with 'train' and 'test'
        small_ds = ds.train_test_split(test_size=frac, seed=seed)["test"]
        small_splits[split_name] = small_ds
    return DatasetDict(small_splits)


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from arc_tartiflette.utils import load

    dataset = {"0a938d79":{"train": [{"input": [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0]]}, {"input": [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4]]}], "test": [{"input": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0]]}]}}
    # dataset_dict = load_dataset("meo-des/arc_main_fmt_aug")
    # dataset_dict = DatasetDict({"train": dataset_dict["train"].select(range(1))})
    dataset_dict = DatasetDict({"train": load.dict_to_transformers_dataset(dataset),})
    tokenizer = AutoTokenizer.from_pretrained("meo-des/nemo_arc_main_base_1s2e_m")

    tokenized_datasets = tokenize_dataset_2DPE(dataset_dict, tokenizer)
    # grid_output = tokenize_example_2DPE(
    #     dataset["0a938d79"]["train"][0], tokenizer, input_beg_coord=(5,10), digit_mapping=get_token_mapping(tokenizer)
    # )
    # print_grid_from_pos_ids(grid_output[0], grid_output[1], tokenizer)
    #
    # print(grid_output)
    # print(tokenizer.decode(3))
    # print(f"Row end id: {get_architects_prompt_format(tokenizer)['row_end_id']}")
    #
    print(tokenized_datasets['train'][0])
    print_grid_from_pos_ids(tokenized_datasets['train'][0]['input_ids'], tokenized_datasets['train'][0]['position_ids'], tokenizer)