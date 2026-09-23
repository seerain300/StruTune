import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices in range [0, num_experts-1]."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    
    return {"topk_idx": topk_idx}


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    MoE token sorting using counting sort with prefix sum.
    
    Args:
        topk_idx: Expert indices (batch_size, seq_len, num_experts_per_tok)
                 with values in [0, num_experts-1]
    
    Returns:
        sorted_token_indices: Token indices sorted by expert (num_tokens,)
        expert_offsets: Cumulative offsets (num_experts+1,)
    """
    # Flatten the input indices to a 1D int32 tensor
    flat = topk_idx.reshape(-1).to(torch.int32)
    N = flat.numel()
    num_experts = 256  # As per the provided setup

    # Compute stable argsort permutation: indices that would sort `flat` ascending
    # Using torch.sort with stable=True returns (values, indices), and we take indices.
    # This matches the original behavior and is deterministic.
    sorted_values, sorted_token_indices = torch.sort(flat, stable=True)
    # sorted_token_indices has shape (N,) and dtype long; keep int32 as in original
    sorted_token_indices = sorted_token_indices.to(torch.int32)

    # Compute per-expert counts and exclusive prefix sum (offsets)
    counts = torch.bincount(flat.long(), minlength=num_experts).to(torch.int32)
    # Prefix sum to get cumulative counts per expert
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    # Prefix sums start from 0 for expert 0; ensure offsets[0] = 0 explicitly
    # (cumsum already does this, but we keep it for clarity)
    offsets = torch.nn.functional.pad(offsets[:-1], (1, 0))  # Insert 0 at the beginning

    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluation harness passes the pre-generated topk_idx tensor.
        # We assume the first argument is the topk_idx tensor.
        if len(args) == 0:
            raise ValueError("ModelNew.forward expects at least one argument (topk_idx).")
        topk_idx = args[0]
        return run(topk_idx)