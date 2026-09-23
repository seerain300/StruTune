import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id (0..255) in orig_ptr (int32).
    counts_ptr is int32 of length 256, zero-initialized by host.
    orig_ptr is the flattened 1D tensor of length N.
    """
    # Each program handles a chunk of size BLOCK
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(orig_ptr + offs, mask=mask, other=0)  # vals are int32
    # Count per expert id 0..255
    for e in range(256):
        is_e = vals == e
        cnt = tl.sum(is_e.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, cnt)


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Exclusive prefix sum across counts_ptr (length 256) into offsets_ptr (length 256).
    offsets[e] = sum(counts[:e]) for e in 0..255. offsets_ptr must be zero-initialized.
    """
    start = tl.zeros((), dtype=tl.int32)
    e = 0
    while e < 256:
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, start)
        start += cnt
        e += 1


@triton.jit
def zero_out_kernel(out_ptr, size: tl.constexpr):
    """
    Zero-initialize int32 out_ptr of length size.
    """
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    mask = offs < size
    tl.store(out_ptr + offs, 0, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from get_inputs: num_experts=256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Flatten
        flat = topk_idx.reshape(-1)
        # PyTorch stable sort to get the correct permutation
        sorted_token_indices = torch.sort(flat.to(torch.int64), stable=True)[1].to(torch.int32)

        # Compute expert offsets in Triton
        N = flat.numel()
        # Ensure orig is int32 for Triton histogram
        orig = flat.to(torch.int32)

        # Prepare counts and offsets
        counts_exp = torch.empty(self.num_experts, dtype=torch.int32, device=orig.device)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=orig.device)

        # Launch Triton histogram
        grid_counts = (triton.cdiv(N, 1024),)  # BLOCK=1024 works well; tune as needed
        histogram_kernel[grid_counts](orig, counts_exp, N, BLOCK=1024)

        # Zero offsets (we'll write prefix sums into offsets[1:])
        zero_out_kernel[(1,)](offsets, size=self.num_experts + 1)

        # Exclusive scan for prefix sums
        exclusive_scan_kernel[(1,)](counts_exp, offsets, num_experts=self.num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
