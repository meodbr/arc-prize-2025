from typing import Iterable

import torch

def pad_and_truncate(tensor: torch.Tensor, max_length: int = -1, pad_value: torch.Tensor | int = 0) -> torch.Tensor:
    """
    Pads and truncates a tensor to a specified maximum length.

    Args:
        tensor (torch.Tensor): The input tensor of shape (L,) where L is the sequence length.
        max_length (int): The desired maximum length of the output tensor.
        pad_value (torch.Tensor | int, optional): The value to use for padding. Defaults to 0.
    Returns:
        torch.Tensor: The padded and truncated tensor of shape (max_length,).
    """
    if isinstance(pad_value, int):
        pad_tensor = torch.tensor(pad_value, dtype=tensor.dtype, device=tensor.device)
    else:
        pad_tensor = pad_value
    
    if max_length <= 0:
        max_length = tensor.size(0)
    
    length = min(tensor.size(0), max_length)

    padded_tensor = pad_tensor.unsqueeze(0).repeat(max_length, 1)
    padded_tensor[:length] = tensor[:length]

    return padded_tensor

def pad_and_truncate_batch(batch: Iterable[torch.Tensor], max_length: int = -1, pad_value: torch.Tensor | int = 0) -> torch.Tensor:
    """
    Pads and truncates a batch of tensor to a specified maximum length.

    Args:
        batch (torch.Tensor): The input tensor of shape (B, L,) where B is the batch size and L is the sequence length.
        max_length (int): The desired maximum length of the output tensor.
        pad_value (torch.Tensor | int, optional): The value to use for padding. Defaults to 0.
    Returns:
        torch.Tensor: The padded and truncated tensor of shape (B, max_length,).
    """
    if isinstance(pad_value, int):
        pad_value = torch.tensor(pad_value, dtype=batch.dtype, device=batch.device)
    
    if max_length <= 0:
        max_length = max(tensor.size(0) for tensor in batch)
    
    padded_batch = torch.full((len(batch), max_length), pad_value, dtype=batch.dtype, device=batch.device)
    for i, tensor in enumerate(batch):
        length = min(tensor.size(0), max_length)
        padded_batch[i, :length] = tensor[:length]
    return padded_batch
