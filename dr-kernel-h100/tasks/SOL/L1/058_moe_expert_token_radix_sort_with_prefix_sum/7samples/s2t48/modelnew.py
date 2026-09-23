import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements and atomically increments counts[id]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)  # int32
    ids = x_vals % E  # expert ids in [0, E)
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def compute_partial_sums(counts_ptr, partial_sums_ptr, carries_ptr, E, BLOCK: tl.constexpr):
    # Sequential scan over counts in tiles of BLOCK; write per-tile sum and carry
    carry = tl.zeros((), dtype=tl.int32)
    for i in range(0, E, BLOCK):
        acc = tl.zeros((), dtype=tl.int32)
        for j in range(0, BLOCK):
            idx = i + j
            # Load count for idx (scalar); guard with mask; idx < E always true here
            val = tl.load(counts_ptr + idx)
            acc += val
        # Store per-tile sum and carry for this tile
        tl.store(partial_sums_ptr + i // BLOCK, acc)
        tl.store(carries_ptr + i // BLOCK, carry)
        carry += acc


@triton.jit
def finalize_offsets(partial_sums_ptr, carries_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix of carries to get carry_inclusive
    # Then offsets[e+1] = offsets[e] + counts[e] via carry_inclusive
    # We iterate across blocks sequentially
    offsets_ptr[0] = 0
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, E, BLOCK):
        acc = tl.zeros((), dtype=tl.int32)
        for j in range(0, BLOCK):
            idx = i + j
            val = tl.load(counts_ptr + idx)
            acc += val
        carry = tl.load(carries_ptr + i // BLOCK)
        total += carry
        tl.store(offsets_ptr + idx, total)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: out receives permutation of positions.
    # starts is exclusive prefix per expert: starts[e] = sum_{j < e} counts[j]
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)
        out_pos = tl.load(starts_ptr + id)
        # Write pos into out at index out_pos; this is a scalar store
        # But Triton kernels don't support dynamic global stores like this directly;
        # instead we implement the logic by reading starts and then updating starts by atomics.
        # We need to update starts[id] += 1 after the write. To do that, we must write to out via a different approach.
        # Therefore, we restructure: for each pos, compute id, find start = starts[id], write pos at out[start], then starts[id] += 1.
        # Since we can't perform arbitrary global stores, we use a vectorized per-iteration approach by reloading x and using masks.
        # However, Triton doesn't support dynamic indexing of out_ptr directly in this manner. To maintain correctness and Triton-only,
        # we restructure the kernel to process positions sequentially and update starts via atomic_add.
        # Simpler approach: We rely on the host to manage out updates using starts; Triton can only do atomic adds.
        # Hence, we restate: for each pos, id = x[pos]; then out[starts[id]] = pos and starts[id] += 1.
        # Implement by reloading x[pos] and using starts[id] for write. Triton allows scalar loads/stores per iteration.
        id_val = tl.load(x_ptr + pos)  # scalar load
        current_start = tl.load(starts_ptr + id_val)  # scalar
        # out_ptr is write-only here; Triton supports scalar store
        tl.store(out_ptr + current_start, pos)
        # update starts[id] += 1
        tl.atomic_add(starts_ptr + id_val, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_hist: int = 1024, block_scan: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.block_hist = block_hist
        self.block_scan = block_scan

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx is the input tensor from get_inputs; ensure it's on CUDA
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device for Triton kernels")
        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Triton histogram: counts of expert ids
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        grid_hist = (triton.cdiv(N, self.block_hist),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=self.block_hist)

        # 2) Compute partial sums and carries for offsets via Triton
        partial_sums = torch.empty(triton.cdiv(E, self.block_scan), dtype=torch.int32, device=x.device)
        carries = torch.empty(triton.cdiv(E, self.block_scan), dtype=torch.int32, device=x.device)
        grid_scan = (1,)
        compute_partial_sums[grid_scan](counts, partial_sums, carries, E, BLOCK=self.block_scan)

        # 3) Finalize offsets: inclusive prefix of carries added to counts
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive prefix starts at 0 for expert 0
        grid_finalize = (1,)
        finalize_offsets[grid_finalize](partial_sums, carries, offsets, E, BLOCK=self.block_scan)

        # 4) Stable counting sort using Triton
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert; starts[e] = offsets[e]
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        return out, offsets