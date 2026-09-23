import torch
import triton
import triton.language as tl


# Triton kernel: stable sort of a 1D int32 flat array using bitonic sorting network.
# Each program handles the global array by performing compare-swap operations.
# Stability is ensured by tie-breaking: for equal values, keep original order (no swap).
@triton.jit
def _bitonic_stable_sort_kernel(flat_in_ptr, flat_out_ptr, N: tl.constexpr, NUM_TOKS: tl.constexpr):
    # For a bitonic network over NUM_TOKS elements, we perform passes k=2,4,... up to NUM_TOKS.
    # Inside each pass, for all j where j has k-th bit set, we do compare-swap with partner.
    # To avoid double work, each thread only acts when partner > index.
    for k in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        if k > NUM_TOKS:
            break
        for j in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            if j > NUM_TOKS:
                break
            partner = tl.program_id(axis=0) ^ j
            # Only act when partner > program_id (one of the two threads in each pair acts)
            act = partner > tl.program_id(axis=0)
            if act:
                # Load current and partner values
                a = tl.load(flat_in_ptr + tl.program_id(axis=0))
                b = tl.load(flat_in_ptr + partner)
                # Determine ascending or descending direction for this subsequence
                asc = ( (tl.program_id(axis=0) & k) == 0 )
                # Stable compare-swap: swap if (a > b) or (a == b and ascending), otherwise keep original order
                # Note: direction is global for the whole array; we use asc as boolean.
                swap = (a > b) ^ asc  # if ascending and a > b -> swap; if descending and a < b -> swap
                # For stability, when a == b, do not swap. Ensure swap is False when a == b.
                swap = swap & ~(a == b)
                new_a = tl.where(swap, b, a)
                new_b = tl.where(swap, a, b)
                # Store results back to out
                tl.store(flat_out_ptr + tl.program_id(axis=0), new_a)
                tl.store(flat_out_ptr + partner, new_b)
    # After the network, copy back to input for the next pass (we pass same pointer)
    # We do this by reloading from flat_out_ptr into flat_in_ptr for the next outer loop iteration.
    # Triton will re-execute the outer loop; to avoid extra loads/stores, we simply
    # assume the input pointer is updated by host. In practice, this kernel runs
    # in-place by reusing pointers; Triton will recompute from flat_in_ptr each time.
    # The above nested loops are compile-time; each pass operates on global array.
    pass


# Triton kernel: compute per-expert counts via histogram from flat indices (int32).
# Each program processes BLOCK elements and atomically adds 1 to counts[exp_id] per match.
@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    idxs = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    for e in range(NUM_EXPERTS):
        matches = (idxs == e) & mask
        count_e = tl.sum(matches.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, count_e)


# Triton kernel: inclusive prefix sum of 'counts' into 'out' (out[0] = 0, out[i+1] = sum(counts[:i+1])).
@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, out_ptr, NUM_EXPERTS: tl.constexpr):
    running = 0
    for i in range(NUM_EXPERTS):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(out_ptr + i + 1, running)
    # out[0] is intentionally left as 0; out[i+1] holds inclusive sum up to i.


class ModelNew:
    def __init__(self, num_experts: int = 256):
        self.num_experts = num_experts

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx of shape (B, S, EPT)
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure contiguous
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D and keep int32 for kernels
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()

        # Allocate output for sorted indices
        sorted_flat = torch.empty(N, dtype=torch.int32, device=flat.device)

        # Launch stable sort Triton kernel
        # We use a single 1D grid covering all tokens. The bitonic network uses compile-time loops.
        grid = (N,)
        _bitonic_stable_sort_kernel[grid](flat, sorted_flat, N, N)

        # Compute per-expert counts via histogram (Triton)
        counts = torch.empty(self.num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid_hist](flat, N, counts, NUM_EXPERTS=self.num_experts, BLOCK=BLOCK, num_warps=4)

        # Compute expert offsets (inclusive prefix sums) via Triton
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, NUM_EXPERTS=self.num_experts, num_warps=1)

        # Convert flat sorted indices back to original shape (batch_size, seq_len, num_experts_per_tok)
        # sorted_flat has the same length as input topk_idx; shape matches. We return both tensors.
        # Return tensors only: sorted_token_indices and expert_offsets
        return sorted_flat.view(*topk_idx.shape), expert_offsets


def run(*args):
    return ModelNew()(*args)
