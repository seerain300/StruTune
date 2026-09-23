import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices(a_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # 1D grid over original indices, process BLOCK indices per program to reduce launch count.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Initialize ranks for the current block to zeros (int32)
    ranks = tl.zeros([BLOCK], dtype=tl.int32)

    # For each original index j, compute how many elements less than a[i] and stable ties.
    # We do this in BLOCK chunks to minimize the number of passes.
    for j in range(0, N, BLOCK):
        joffs = j + tl.arange(0, BLOCK)
        jmask = joffs < N

        # Load values for j-offs
        aj = tl.load(a_ptr + joffs, mask=jmask, other=tl.zeros((), dtype=tl.int32))

        # For each i in the current program, compute stable rank contribution from aj
        # Note: we accumulate ranks only for valid i (offs < N)
        for k in range(0, BLOCK):
            i = offs[k]
            valid_i = i < N
            # Load ai; if i is out of range, skip
            ai = tl.load(a_ptr + i, mask=valid_i, other=0)

            # Compare against all aj
            # We compute less and tie for all valid j; masked by jmask
            less = tl.zeros((), dtype=tl.int32)
            tie = tl.zeros((), dtype=tl.int32)
            for kk in range(0, BLOCK):
                jj = joffs[kk]
                valid_j = jj < N
                ajk = tl.load(a_ptr + jj, mask=valid_j, other=tl.zeros((), dtype=tl.int32))
                less += ((ajk < ai) & valid_j).to(tl.int32)
                tie += ((ajk == ai) & (jj < i) & valid_j).to(tl.int32)

            # Only update ranks for valid i
            ranks += (less + tie) * valid_i

    # Write out the permutation: index i goes to position ranks[i]
    # We write only for valid i (offs < N)
    for k in range(0, BLOCK):
        i = offs[k]
        valid_i = i < N
        tl.store(out_ptr + ranks[k], i, mask=valid_i)


@triton.jit
def _histogram_kernel(a_ptr, N: tl.constexpr, hist_ptr, num_buckets: tl.constexpr):
    # Each element performs one atomic add into its bucket
    for i in range(0, N):
        val = tl.load(a_ptr + i)
        # val is int32 in [0, num_buckets-1]
        tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(in_vec_ptr, out_ptr, num_buckets: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_buckets):
        acc += tl.load(in_vec_ptr + i)
        tl.store(out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D and ensure int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()
        device = flat.device
        num_experts = 256  # matches original run

        # Allocate output permutation (length N)
        out = torch.empty(N, dtype=torch.int32, device=device)

        # Launch Triton stable argsort kernel
        BLOCK = 256  # process 256 indices per program
        grid = (triton.cdiv(N, BLOCK),)
        _stable_argsort_indices[grid](flat, out, N, BLOCK)

        # Histogram of expert IDs using Triton
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_experts)

        # Prefix sum to get expert_offsets of length (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
