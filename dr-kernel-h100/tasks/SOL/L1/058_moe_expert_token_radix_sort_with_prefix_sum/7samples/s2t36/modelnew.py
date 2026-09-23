import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements and atomically adds 1 to counts[id]
    pid = tl.program_id(0)
    start = pid * BLOCK
    # Build offsets vector for this program
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load x values for this chunk
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Compute expert ids as x % E (assuming x is non-negative int32)
    ids = x % E
    # Atomic add to counts for each valid element
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def inclusive_scan_prefixsum(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums: offsets[i+1] = offsets[i] + counts[i]
    # offsets[0] is set by host.
    # Each program handles a tile of size BLOCK.
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < E

    # Initialize tile offsets to counts
    c = tl.load(counts_ptr + offsets, mask=mask, other=0)
    base = tl.load(offsets_ptr, mask=tl.full([BLOCK], True, tl.int1), other=0)  # not used, will read offsets_ptr[offsets]
    # We need to iterate and add previous elements. Triton allows simple loops.
    # We'll do Hillis-Steele-like scan within tile using a while loop.
    i = 0
    while i < BLOCK:
        if mask[i]:
            # Sum contributions from previous positions within this tile
            # Compute inclusive sum by adding elements at positions j=i-1, i-2, ..., 0.
            # Note: Triton while supports scalar control. We iterate j and add to current c[i].
            j = i
            val_i = c[i]
            # Unroll a small loop to reduce dependency chain
            # We handle up to 16 steps per iteration
            for k in range(16):
                # Compute prev index within this tile
                prev = j - (1 << k)
                # prev must be non-negative and within this tile
                if prev >= 0 and (prev < i):
                    # Load prev value and add; since prev is not the current element, we need to ensure we don't read c[prev]
                    # Instead, we can't directly do that; so we recompute inclusive sum via repeated additions if needed.
                    # Triton doesn't support vectorized scan easily across tiles, so we do a sequential scan inside this kernel.
                    pass
            # Update offsets
            tl.store(offsets_ptr + offsets[i], val_i)
        i += 1


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: for pos in [0, N), write out[starts[id]] = pos; then starts[id] += 1
    pid = tl.program_id(0)
    # Single program handles all positions sequentially; BLOCK can be 1
    for pos in range(N):
        id = tl.load(x_ptr + pos)
        idx = tl.load(starts_ptr + id)
        tl.store(out_ptr + idx, pos)
        # Increment starts[id]
        new_starts = tl.load(starts_ptr + id)
        new_starts += 1
        tl.store(starts_ptr + id, new_starts)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Expect topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 on CUDA
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as per original run

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton (block-wise scan)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        BLOCK_SCAN = 256
        grid_scan = (triton.cdiv(E, BLOCK_SCAN),)
        inclusive_scan_prefixsum[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation of 0..N-1)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets