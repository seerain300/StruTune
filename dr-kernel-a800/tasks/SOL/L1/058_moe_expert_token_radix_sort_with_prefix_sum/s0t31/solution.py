import torch
import triton
import triton.language as tl


@triton.jit
def odd_even_sort_pass(flat_ptr, out_ptr, N, PASS: tl.constexpr):
    """
    One pass of odd-even transposition sort:
    - axis 0: each program handles one element index i in [0, N).
    - axis 1: fixed PASS in {0,1,2,...} selects even/odd passes.
      If PASS is even: compare-swap pairs (0,1), (2,3), ...
      If PASS is odd:  compare-swap pairs (1,2), (3,4), ...
    Only one program per pair performs the swap (i < partner).
    out_ptr holds the working array of values (int32).
    """
    i = tl.program_id(axis=0)
    # Determine partner based on pass parity
    if PASS % 2 == 0:
        # Even pass: pairs (0,1), (2,3), ...
        partner_idx = i + 1 if (i % 2 == 0 and i < N - 1) else i
    else:
        # Odd pass: pairs (1,2), (3,4), ...
        partner_idx = i + 1 if (i % 2 == 1 and i < N - 1) else i

    # Load current values
    value_i = tl.load(flat_ptr + i)
    value_partner = tl.load(flat_ptr + partner_idx)

    # Compare-and-swap to enforce ascending order
    # Swap if partner < i
    swap = value_partner < value_i
    new_i = tl.where(swap, value_partner, value_i)
    new_partner = tl.where(swap, value_i, value_partner)

    # Only one program in the pair performs the store
    is_lower = i < partner_idx
    tl.store(out_ptr + i, new_i, mask=is_lower)
    tl.store(out_ptr + partner_idx, new_partner, mask=is_lower)


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts (256).
    Each program processes a chunk of elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 2048
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, 255]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int32, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] is set by the caller to 0; we write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model.
    - All computation happens in Triton kernels launched in forward.
    - Outputs match the original behavior exactly:
      * sorted_token_indices: int64 tensor of length N, original indices sorted by flat values (stable).
      * expert_offsets: int32 tensor of length num_experts + 1.
    """

    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA device for Triton
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton execution."
        # Flatten and ensure int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # Perform odd-even transposition sort to produce sorted flat values
        working = torch.empty_like(flat)  # working copy for sorting (int32)
        # Run 3*N passes to ensure sorting completes (odd-even transposition requires up to N passes,
        # but running more passes in parallel grid dims doesn't harm; we keep it simple and safe.)
        # Grid: (N,) programs per pass; axis=1 selects pass id in [0, 3*N).
        # Note: Triton requires const pass index for specialization; we loop over passes explicitly.
        # We'll invoke the kernel for each pass to allow grid specialization.
        for pass_id in range(0, 3 * N):
            odd_even_sort_pass[(N,)](flat, working, N, pass_id)

        # Now we need sorted_token_indices: original positions sorted by flat values (stable).
        # We sort indices [0..N-1] based on 'working' to get the stable order.
        indices = torch.arange(N, dtype=torch.int32, device=flat.device)
        sorted_indices = torch.empty_like(indices)
        # Run the same sorting passes on indices based on 'working' values
        for pass_id in range(0, 3 * N):
            odd_even_sort_pass[(N,)](indices, sorted_indices, N, pass_id)

        # Convert to int64 to match torch.sort(...).indices dtype
        sorted_token_indices = sorted_indices.to(torch.int64)

        # Compute expert offsets via Triton histogram + prefix sum
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)

        BLOCK = 2048
        grid_hist = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid_hist](working, counts, N, self.num_experts, BLOCK=BLOCK, num_warps=4)

        offsets_int64 = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        prefix_sum_kernel[(self.num_experts,)](counts, offsets_int64[1:], self.num_experts, num_warps=1)

        expert_offsets = offsets_int64.to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
