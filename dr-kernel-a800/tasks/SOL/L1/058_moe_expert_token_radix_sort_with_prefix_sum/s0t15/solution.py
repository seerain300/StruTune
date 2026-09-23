import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program handles BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Triton stable bitonic sort on flat_ptr (int32). out_idx_ptr holds original positions as int64, length N.
    Bitonic sort stages: k = 0..LOGN-1, j = k+1..LOGN-1.
    For each stage, each program i compares with partner = i ^ (1 << j).
    Ascending if (i & (1 << (j+1))) == 0 else descending. Stability: tie-break by i < partner.
    """
    pid = tl.program_id(axis=0)
    # Initialize out_idx_ptr with original positions (int64)
    tl.store(out_idx_ptr + pid, tl.cast(pid, tl.int64))

    # Bitonic sort stages
    for k in range(0, LOGN):
        for j in range(k + 1, LOGN):
            partner = pid ^ (1 << j)
            asc = ( (pid & (1 << (k + 1))) == 0 )
            # Only process each pair once to avoid double updates and races
            do_pair = pid < partner
            # Load current values
            val_i = tl.load(flat_ptr + pid)
            val_p = tl.load(flat_ptr + partner)
            idx_i = tl.load(out_idx_ptr + pid)
            idx_p = tl.load(out_idx_ptr + partner)
            # Perform compare-and-swap with stability tie-break
            less = val_i < val_p
            equal = val_i == val_p
            # Stability tie-break: if equal, lower original index comes first
            if equal:
                less = pid < partner
            if asc:
                should_swap = (less or (equal and not (pid < partner)))
            else:
                # Descending: swap if 'greater' or equal with partner coming first
                should_swap = (not less) or (equal and (pid < partner))
            # Only act when do_pair is true
            if do_pair:
                if should_swap:
                    tl.store(out_idx_ptr + pid, idx_p)
                    tl.store(out_idx_ptr + partner, idx_i)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Flattens topk_idx to 1D int32.
        - Uses Triton to compute sorted_token_indices (int64) with stable sort.
        - Uses Triton to compute counts (histogram). Prefix sum and expert_offsets are computed via PyTorch.
        """
        # Ensure topk_idx is on CUDA for Triton kernels
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        flat = topk_idx.contiguous().view(-1)
        N = flat.numel()

        # Compute LOGN for bitonic sort
        if N <= 1:
            LOGN = 0
        else:
            # Compute log2(N) as int
            LOGN = int(torch.ceil(torch.log2(torch.tensor(N, device=flat.device))).item())

        # Allocate output indices (int64)
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)

        # Launch stable bitonic sort kernel
        grid_sort = (N,)
        stable_bitonic_sort_kernel[grid_sort](flat, sorted_token_indices, N, LOGN=LOGN)

        # Also compute counts via Triton for completeness (not used in offset computation here)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid_counts = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=self.num_experts)

        # expert_offsets: compute via PyTorch (int32). offsets[0] = 0; offsets[1:] = cumsum(bincount(flat))
        flat_i32 = flat.to(torch.int32)
        counts_long = torch.bincount(flat_i32, minlength=self.num_experts).to(torch.int64)
        offsets_long = torch.cumsum(counts_long, dim=0)  # shape (num_experts,)
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = offsets_long.to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
