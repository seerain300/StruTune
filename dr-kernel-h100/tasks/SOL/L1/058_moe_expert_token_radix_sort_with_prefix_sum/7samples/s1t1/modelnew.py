import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-expert counts via one atomic add per token.
# We process tokens in vector chunks to improve throughput.
@triton.jit
def _count_experts_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Program id: 0 for now (grid size 1). We iterate over the input in chunks.
    # Each thread lane processes a vector of indices in a loop over chunk_start.
    # Use a compile-time unrolled loop over chunks by passing NUM_ITERS as constexpr if desired,
    # but here we keep it simple and rely on while-loop with static block-size.
    num_threads = tl.num_programs(0)  # not used; Triton doesn't support tl.num_threads, we use a single grid
    # Instead, we use a single program instance and loop over the entire vector:
    # This kernel is intended to run with grid=1. To improve parallelism, we can split across chunks.
    # However, to simplify and ensure correctness, we use grid=1 and loop over N in BLOCK_SIZE chunks.
    # Note: Triton requires grid to be 1 for this design; adjust grid to cover all tokens in Python.
    pass  # This is a placeholder; see below for the real implementation with grid.


# Simpler, effective Triton kernel that counts per token with a loop over chunks.
# We'll run it with grid=1, and loop across N in chunks of BLOCK_SIZE.
@triton.jit
def _count_experts_atomic_kernel_loop(flat_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single program instance processes the entire array in chunks.
    # We use a while loop with static BLOCK_SIZE for vectorized loads.
    start = 0
    offsets = tl.arange(0, BLOCK_SIZE)
    while start < N:
        idx = start + offsets
        mask = idx < N
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # int32
        # For masked elements, vals is 0; tl.load with other=0 ensures valid value.
        # Atomic add 1 for each valid element.
        # Triton allows pointer arithmetic on int32 addresses; counts_ptr is int32*.
        # Cast vals to int32 if needed.
        # Note: atomic_add expects int32 values; each token contributes +1 regardless of expert.
        # We need to add 1 for each valid token, not per expert; but per-expert counting is done inside.
        # Therefore, we perform per-expert counting using a separate approach: per-thread vectorized adds.
        # The below is a conceptual implementation; in practice, we'll use per-thread vectorized adds
        # by launching the kernel with multiple threads per program instance.
        # Triton does not support dynamic grid creation with loops here, so we structure it differently:
        # Use a kernel that each program instance handles a chunk.
        pass  # Placeholder; see the actual kernel below.


# Actual Triton kernel: each program instance handles a chunk of tokens and performs atomics per token.
@triton.jit
def _count_experts_atomic_kernel_grid(flat_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # For each valid token in this chunk, perform atomic add to counts[vals].
    # This avoids a host-side loop and leverages parallelism across program instances.
    for i in range(BLOCK_SIZE):
        if mask[i]:
            idx = offsets[i]
            val = vals[i]  # expert index
            # Atomic add one to counts[val]
            # In Triton, atomic_add on pointers requires int32 and supports +=.
            # counts_ptr is a base pointer; add val to index by pointer arithmetic.
            # counts_ptr + val points to the count for that expert.
            # Increment by 1.
            tl.atomic_add(counts_ptr + val, 1)


# Kernel 2: Compute inclusive prefix sums of counts into offsets.
# Since num_experts=256 is small, a simple per-expert loop in a single program instance is fine.
@triton.jit
def _prefix_sum_inclusive_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # offsets_ptr[0] = 0 (implicitly since we write from 1)
    acc = 0
    for e in range(0, num_experts):
        # Load count for expert e
        c = tl.load(counts_ptr + e)  # int32
        acc += c
        # Write inclusive sum
        tl.store(offsets_ptr + e + 1, acc)
    # offsets_ptr[num_experts] remains as the sum of all counts (implicit last store at e=num_experts-1).
    # We can set the last element to N for correctness if we prefer, but it will be acc after the loop.
    # To be explicit, the last inclusive sum should be total N. Since sum of counts equals N, acc is N.
    # So no additional write needed.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (B, S, M), int32, on CUDA
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        flat = topk_idx.reshape(-1)  # 1D int32, length N
        # We will compute counts via Triton, then sort with torch, then prefix sum with Triton.
        N = flat.numel()
        num_experts = 256

        # 1) Triton kernel: per-expert counts via atomics (vectorized across chunks).
        # Allocate counts vector of length num_experts, initialized to zeros.
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Choose a reasonable BLOCK_SIZE; 1024 works well for moderate N.
        BLOCK_SIZE = 1024
        # Launch grid across chunks: number of program instances equals ceil_div(N, BLOCK_SIZE).
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _count_experts_atomic_kernel_grid[grid](flat, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Inclusive prefix sums of counts into offsets (num_experts + 1).
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # offsets[0] will be 0; we write from index 1 onward in the kernel.
        _prefix_sum_inclusive_kernel[(1,)](counts, offsets, num_experts=num_experts, num_warps=1)

        # 3) Stable sort to get the permutation of token indices that sorts by expert IDs.
        #    We keep torch.sort here for correctness and performance.
        # Note: flat is non-differentiable; we don't need gradients.
        # stable=True ensures tie-breaking behavior matches original code.
        _, sorted_token_indices = flat.sort(stable=True)

        # Return as original API: sorted_token_indices (int32), expert_offsets (int32)
        # sorted_token_indices is already int64 from torch.sort; cast to int32 to match original.
        return sorted_token_indices.to(torch.int32), offsets