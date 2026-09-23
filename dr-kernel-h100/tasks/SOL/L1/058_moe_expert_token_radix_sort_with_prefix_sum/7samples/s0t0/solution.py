import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount(input_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute histogram of int32 values in input_ptr[0:N] into counts_ptr[0:256].
    counts_ptr is int32, input_ptr is int32.
    Each program instance processes BLOCK_SIZE elements of the input vector.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load values. Provide 0 for out-of-bounds; masked store will ignore them anyway.
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 for each valid value into counts[vals]
    # Note: counts_ptr is int32 and we add int32 ones.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT), int32, CUDA tensor
        Returns:
          sorted_token_indices: int32 tensor of shape (N,), permutation [0..N-1] sorted by flattened values.
          expert_offsets: int32 tensor of shape (257,), cumulative counts per expert including prefix.
        """
        # Ensure contiguous and flatten
        flat = topk_idx.reshape(-1).contiguous()

        # Triton bincount: histogram of values in flat (int32 -> int32 counts)
        N = flat.numel()
        # We assume num_experts is 256 as in the original reference code
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel
        # BLOCK_SIZE can be tuned; 1024 or 2048 are good defaults
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        triton_bincount[grid](flat, counts, N, BLOCK_SIZE=BLOCK_SIZE)

        # expert_offsets: inclusive cumsum at index i gives total tokens assigned to expert i
        # Original code returns length num_experts+1 with the 0-based prefix, but returns cumsum, not exclusive.
        # We will add a zero at index 0, then append zeros for indices >= number of unique experts (not needed here since we pad to 256)
        # However, because we padded to 256, and counts are zeros for missing ids, cumsum will be correct.
        # To match the original behavior, we pad to minlength=num_experts (256) and take cumsum.
        # torch.cumsum expects int64 for counts input; cast counts to int64 for cumsum
        counts64 = counts.to(torch.int64)
        # Create expert_offsets of length 257 (0..256)
        expert_offsets = torch.cumsum(counts64, dim=0)  # length 256
        # Add leading zero
        expert_offsets = torch.nn.functional.pad(expert_offsets, (1, 0), mode='constant', value=0)

        # sorted_token_indices: stable argsort of flattened indices by values. Keep torch.sort for correctness and performance.
        # torch.sort requires floating or integral types; int64 is fine
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
