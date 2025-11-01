import torch
from transformers import AutoTokenizer
from datasets import DatasetDict

from arc_tartiflette.config.settings import ENV_VARS


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
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print("---- Dataset tokenized. ----")
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")
    return tokenized_datasets


def compute_2DPE_pos_ids(example, tokenizer: AutoTokenizer):
    """
    Computes 2D positional IDs for a given example.

    Args:
        example (dict): Example with 'grid_width' and 'grid_height' keys.

    Returns:
        torch.Tensor: [batch, seq_len] 2D positional IDs.
    """
    grid_width = example.get("grid_width", 1)
    grid_height = example.get("grid_height", 1)
    pos_ids = torch.zeros((grid_height, grid_width), dtype=torch.long)

    for i in range(grid_height):
        for j in range(grid_width):
            pos_ids[i, j] = i * grid_width + j

    return pos_ids


def tokenize_dataset_2DPE(dataset_dict: DatasetDict, tokenizer: AutoTokenizer):
    print("Tokenizing dataset with 2DPE tokenizer...")
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
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
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
