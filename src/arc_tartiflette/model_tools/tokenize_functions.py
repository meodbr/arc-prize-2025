import torch

def make_completion_mask(input_ids: torch.Tensor, special_token_id: int, n: int) -> torch.Tensor:
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

    return loss_mask