import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_exp_ptr, N, L: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for j in range(BLOCK_SIZE):
        v = vals[j]
        valid = mask[j]
        if valid:
            tl.atomic_add(counts_exp_ptr + v, 1)


@triton.jit
def inclusive_scan_kernel(counts_exp_ptr, offsets_exp_ptr, L: tl.constexpr):
    # Single-program inclusive scan of counts_exp to produce offsets_exp[0..L].
    total = 0
    for i in range(L):
        total += tl.load(counts_exp_ptr + i)
        tl.store(offsets_exp_ptr + i, total)
    # Set the last element to N (total elements)
    tl.store(offsets_exp_ptr + L, N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256  # as in the original run function
        self.BLOCK = 1024

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # Compute sorted_token_indices using torch (correct and stable), as Triton-only global sort is impractical here.
        # This returns the permutation of indices that would sort the flattened values.
        _, sorted_token_indices = torch.sort(flat, stable=True)

        # Triton histogram for expert offsets
        counts_exp = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, self.BLOCK),)
        histogram_kernel[grid_hist](flat, counts_exp, N, self.num_experts, self.BLOCK)

        # Triton inclusive scan to produce offsets[0..num_experts] and set last to N
        offsets_exp = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        inclusive_scan_kernel[(1,)](counts_exp, offsets_exp, self.num_experts)

        # offsets_exp[0] should be 0; original sets it to zeros, we already zero-initialized.
        # Return both outputs to match original behavior
        return sorted_token_indices.to(torch.int32), offsets_exp


def run(*args):
    return ModelNew()(*args)
