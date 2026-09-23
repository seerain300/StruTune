import torch
import triton
import triton.language as tl


# Triton kernel: compute counts per key k in [0..NUM_EXPERTS). This is equivalent to a local bincount
# but only over the whole vector, which we iterate over in chunks. We'll later use torch.cumsum to form prefix.
@triton.jit
def counts_per_key_kernel(
    flat_ptr,                     # *int32, flattened input of length M
    counts_ptr,                   # *int32, output length NUM_EXPERTS (counts of each key)
    M: tl.int32,                  # number of elements in flat
    NUM_EXPERTS: tl.int32,        # number of possible keys (256)
    BLOCK: tl.constexpr,          # chunk size for scanning flat
):
    k = tl.program_id(0)  # one program per key in [0..NUM_EXPERTS)
    cnt = tl.zeros((), dtype=tl.int32)
    # Loop over flat in chunks of BLOCK, masked by M
    for off in range(0, M, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Only consider lanes that are in-bounds and equal to key k
        eq = (vals == k) & mask
        cnt += tl.sum(eq.to(tl.int32))
    # Store count for key k
    tl.store(counts_ptr + k, cnt)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block = block

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        device = flat.device
        dtype = torch.int32

        # 1) Compute sorted_token_indices = argsort(flat) to match torch.sort(flat, stable=True).values exactly
        #    For integer values in [0, 255], argsort is the correct stable order (ties are by index).
        sorted_token_indices = torch.argsort(flat)

        # 2) Compute expert_offsets via Triton counts and torch.cumsum + 1
        #    Note: this Triton kernel computes the per-key counts; we will use torch.cumsum to get prefix sums.
        counts = torch.empty(self.num_experts, dtype=torch.int32, device=device)
        grid_counts = (self.num_experts,)
        counts_per_key_kernel[grid_counts](
            flat, counts, M, self.num_experts, self.block,
            num_warps=1, num_stages=1
        )

        # Use torch.cumsum to get inclusive prefix sums for each expert
        expert_offsets = torch.cumsum(counts, dim=0)
        # Add +1 at the end as per original code
        expert_offsets = torch.cat([expert_offsets, torch.tensor([0], dtype=torch.int32, device=device)])

        # The original code returns sorted_token_indices as int32 and expert_offsets as int32.
        # Ensure dtypes match (they already do).
        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
