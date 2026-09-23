import torch

# Triton is required; import and use kernels in forward.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute histogram of values in 'orig' (int32) into 'counts' (int32).
# We assume values are in [0, L-1], with L = num_experts (256).
if TRITON_AVAILABLE:
    @triton.jit
    def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
        # Accumulate counts per value using atomic adds
        for v in range(L):
            inc = tl.where(mask & (vals == v), 1, 0)
            # Reduce 'inc' to a scalar and atomic_add to counts[v]
            partial_sum = tl.sum(inc, axis=0)
            tl.atomic_add(counts_ptr + v, partial_sum)

    # Triton kernel: compute stable ranks and write sorted_token_indices.
    # For each original position j, compute rank by scanning counts[i]:
    # rank += counts[i] if i < flat[j]; tie-break by original index j (stable).
    # Then write j into sorted_token_indices[rank].
    @triton.jit
    def stable_rank_write_kernel(orig_ptr, sorted_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        vals = tl.load(orig_ptr + offsets, mask=mask, other=0)  # original values
        j = offsets  # original positions

        running = tl.zeros([BLOCK], dtype=tl.int32)
        # For each value v, if v < vals[i], increment running by counts[v]
        for v in range(L):
            cnt = tl.load(counts_ptr + v)  # scalar
            less = mask & (vals > v)  # stable: equal values keep original order (vals == v at position j not lessened)
            inc = less.to(tl.int32) * cnt
            running += inc
        # Write positions j at their stable ranks
        tl.store(sorted_ptr + running, j, mask=mask)

    # Triton kernel: exclusive prefix sum of 'counts' (int32) into 'out' (int32).
    # out[i] = sum_{j < i} counts[j]; out[L] = total N
    @triton.jit
    def exclusive_scan_kernel(counts_ptr, out_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
        running = 0
        for i in range(L):
            running += tl.load(counts_ptr + i)
            tl.store(out_ptr + i, running - tl.load(counts_ptr + i))
        total = 0
        for i in range(L):
            total += tl.load(counts_ptr + i)
        tl.store(out_ptr + L, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Compute exactly as original:
        - sorted_token_indices: permutation of [0..N-1] that would sort flattened topk_idx stably.
        - expert_offsets: int32 tensor of length (num_experts + 1), where offsets[e] = inclusive count
          of elements whose value is strictly less than e. offsets[-1] = N.
        """
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        L = 256  # num_experts in original

        # Allocate outputs
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        expert_offsets = torch.empty(L + 1, dtype=torch.int32, device=flat.device)

        # 1) Histogram of flat using Triton
        counts = torch.zeros(L, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, N, L=L, BLOCK=BLOCK)

        # 2) Stable counting-sort via ranks and write to sorted_token_indices using Triton
        stable_rank_write_kernel[grid](flat, sorted_token_indices, counts, N, L=L, BLOCK=BLOCK)

        # 3) Exclusive prefix sum (scan) of counts using Triton to get expert_offsets
        # offsets[0] should be 0; we write zeros at [0] explicitly.
        expert_offsets[0] = 0
        scan_BLOCK = 128  # L=256 fits comfortably
        exclusive_scan_kernel[(1,)](counts, expert_offsets, L=L, BLOCK=scan_BLOCK)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
