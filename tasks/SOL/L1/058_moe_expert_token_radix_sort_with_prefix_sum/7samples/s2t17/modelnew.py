import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles a chunk of size BLOCK
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load elements; if mask is False, load 0 (won't contribute to counts)
    ids = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Compute expert index and atomic add into counts
    # Note: ids are int32 and in [0, E)
    # We can't index with vector directly; we loop to do atomic_add per element
    for i in range(BLOCK):
        idx = offsets[i]
        if idx < N:
            # ids[i] is the value at idx; ensure int32
            # atomic add to counts[ids[i]]
            tl.atomic_add(counts_ptr + ids[i], 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # In-place block-wise inclusive scan over counts_ptr -> offsets_ptr (length E+1)
    # offsets_ptr[0] must be initialized to 0 on host before launch.
    # Each program handles a tile of size BLOCK starting at pid*BLOCK.
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < E

    # First pass: compute exclusive prefix for each element in the tile
    running = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(BLOCK):
        pos = start + i
        if pos < E:
            val = tl.load(counts_ptr + pos)
            running[i] = val
            tl.store(offsets_ptr + pos + 1, val)  # exclusive prefix for pos is val
        # Update running for j > i using previously stored exclusive prefixes
        # For j > i: running[j] += running[i] if offset[i] is written already
        # We emulate scan via loop:
        for j in range(i + 1, BLOCK):
            posj = start + j
            if posj < E:
                prev = tl.load(offsets_ptr + (posj))  # exclusive prefix for posj before update
                # We cannot directly update across threads; instead, we rely on the fact that
                # running[j] is computed from previous running[j] and add prev for j > i.
                # Since tl.atomic_add cannot be used here, we use a simple trick: write a dummy
                # and rely on out-of-order correctness. For simplicity and correctness, we keep
                # a scalar loop and only update when j > i by adding running[i] if posj >= start+i
                # (since running[i] is already computed). Triton requires scalar-like updates; thus,
                # we implement per-element updates in a scalar loop below instead of vectorized atomic ops.
    # Fall back to a scalar loop per element inside this program for correctness:
    for i in range(BLOCK):
        pos = start + i
        if pos < E:
            val = tl.load(counts_ptr + pos)
            running = val
            tl.store(offsets_ptr + pos + 1, val)  # exclusive prefix for pos
            for j in range(i + 1, BLOCK):
                posj = start + j
                if posj < E:
                    # prev is the exclusive prefix for posj BEFORE including running[j]
                    prev = tl.load(offsets_ptr + posj)
                    # inclusive: prev + running
                    inc = prev + running
                    tl.store(offsets_ptr + posj, inc)
                    # running update for j > i
                    running += tl.load(offsets_ptr + (start + i))  # exclusive prefix at i


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: out contains permutation indices in stable order
    # starts_ptr has exclusive prefix sums per expert. We iterate over positions and write
    # each position to its destination based on id = x[pos].
    # This loop is sequential over positions; for correctness and stability we keep it simple.
    # BLOCK here is just a placeholder; we use scalar loop over N.
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)
        dest = tl.load(starts_ptr + id)
        tl.store(out_ptr + dest, pos)
        # update starts[id]
        old_starts = tl.load(starts_ptr + id)
        new_starts = old_starts + 1
        tl.store(starts_ptr + id, new_starts)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Use provided topk_idx (batch_size, seq_len, num_experts_per_tok), int32, on CUDA.
        - Compute:
          * sorted_token_indices: stable permutation of flattened topk_idx
          * expert_offsets: inclusive prefix sums of counts per expert (length num_experts+1)
        Returns (sorted_token_indices, expert_offsets).
        """
        # Ensure Triton-compatible dtype and device
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts  # 256 as in original code

        # 1) Histogram of expert IDs using Triton (chunked atomic_add)
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048  # larger block improves throughput
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton (block-wise scan)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        BLOCK_SCAN = 256  # tile size for scan
        grid_scan = (triton.cdiv(E, BLOCK_SCAN),)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        return out, offsets