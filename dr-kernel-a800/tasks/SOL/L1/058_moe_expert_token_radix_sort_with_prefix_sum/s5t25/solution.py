import torch
import triton
import triton.language as tl


# Kernel: For each expert e in [0, NUM_EXPERTS), count how many times e appears in flat.
@triton.jit
def counts_kernel(flat_ptr, counts_ptr, M, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)  # one program per expert
    count = tl.zeros((), dtype=tl.int32)
    # Iterate over the flattened vector in chunks of BLOCK
    for start in range(0, M, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        eq = (vals == e) & mask
        count += tl.sum(eq.to(tl.int32), axis=0)
    tl.store(counts_ptr + e, count)


# Kernel: Inclusive prefix sum over counts[0..NUM_EXPERTS-1] -> offsets_incl[0..NUM_EXPERTS-1]
@triton.jit
def scan_inclusive_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # Single-program scan across NUM_EXPERTS in chunks
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < NUM_EXPERTS
        cnts = tl.load(counts_ptr + idx, mask=mask, other=0)  # int32
        running += tl.sum(cnts, axis=0)
        tl.store(offsets_ptr + idx, running, mask=mask)


# Kernel: Finalize expert_offsets: write offsets_incl[:NUM_EXPERTS] and set last = total_count + 1.
@triton.jit
def finalize_offsets_kernel(offsets_incl_ptr, total_count_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # Copy inclusive prefix sums to offsets[:NUM_EXPERTS]
    for e in range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + e, tl.load(offsets_incl_ptr + e))
    # Set last element = total_count + 1
    total = tl.load(total_count_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - sorted_token_indices: torch.long tensor of shape (M,), stable sort indices of the flattened values.
          - expert_offsets: torch.int32 tensor of shape (num_experts+1,), inclusive prefix counts per expert + 1.
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        M = flat.numel()


def run(*args):
    return ModelNew()(*args)
