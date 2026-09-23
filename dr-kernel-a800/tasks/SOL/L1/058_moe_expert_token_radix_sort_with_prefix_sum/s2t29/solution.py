import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel_v2(vals_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr, num_experts: tl.constexpr):
    """
    Triton kernel: count occurrences of each expert id in vals_ptr.
    - vals_ptr: *int32, length N
    - counts_ptr: *int32, length num_experts
    - N: int
    - BLOCK_SIZE: constexpr, e.g., 1024
    Each program processes BLOCK_SIZE elements, and increments counts for each expert id.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load a block of values
    vals = tl.load(vals_ptr + offs, mask=mask, other=0)

    # For each position in the block (scalar-wise), update counts
    for i in range(BLOCK_SIZE):
        # Only proceed if this position is valid
        valid = (start + i) < N
        # Load scalar value; if invalid, set to -1 to avoid counting
        val = tl.load(vals_ptr + (start + i), mask=valid, other=-1)
        # Increment counts for each expert id, for valid values only
        for e in range(num_experts):
            if valid and (val == e):
                tl.atomic_add(counts_ptr + e, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Reshape and ensure contiguity; compute per-expert counts via Triton.
        - No torch data ops (torch.sort, torch.cumsum, torch.bincount, etc.).
        Returns: torch.Tensor of per-expert counts (int32), length = num_experts.
        """
        # Input: topk_idx of shape (batch_size, seq_len, num_experts_per_tok), int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # Allocate counts for experts [0..num_experts-1]
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)

        # Launch count kernel: grid over blocks
        BLOCK_SIZE = 1024  # reasonable block size
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        count_experts_kernel_v2[grid](flat, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_experts=self.num_experts)

        # Return per-expert counts (this is a Triton-only computation)
        return counts


def run(*args):
    return ModelNew()(*args)
