import torch
import triton
import triton.language as tl


@triton.jit
def counts_by_exp_kernel(flat_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.int32):
    # One program per expert id
    e = tl.program_id(0)
    # Loop over flat in chunks of BLOCK
    BLOCK = 1024
    start = 0
    while start < M:
        idx = start + tl.arange(0, BLOCK)
        mask = idx < M
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # vals are int32
        # Check equality for this expert and masked load
        eq = (vals == e) & mask
        # Atomic add 1 for each match
        # Triton's atomic_add expects integer accumulation; cast eq to int32.
        # eq is a vector; atomic_add will handle per-lane independently where mask is True.
        tl.atomic_add(counts_ptr + e, eq.to(tl.int32).sum())
        start += BLOCK


@triton.jit
def cumsum_inclusive_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.int32):
    # Compute inclusive prefix sums into offsets_ptr[0..NUM_EXPERTS-1]
    # offsets_ptr is length NUM_EXPERTS. We'll compute serially in this small vectorized fashion.
    # Each program id 0..NUM_EXPERTS-1 computes its offset as sum(counts[0..pid]).
    # But Triton kernels are parallel; we need a loop. Use tl.static_range if NUM_EXPERTS is constexpr,
    # but here it's runtime. We implement a simple sequential accumulation per output using a temporary.
    # Alternatively, we can do a parallel scan using shared memory; but this kernel is small.
    # We'll implement a per-output serial scan using a loop over i in range(NUM_EXPERTS).
    # Note: Triton supports loops; we use a runtime loop here by computing offsets sequentially.
    # Create a scalar accumulator acc
    acc = 0
    # For each e from 0 to NUM_EXPERTS-1, compute acc += counts[e] and store
    # Since we can't branch on dynamic NUM_EXPERTS easily in Triton, we instead do a vectorized
    # approach: read counts into a vector and compute prefix sums. However, Triton kernel does not
    # have shared scratch, so we implement per-output sequential accumulation by reading counts[e].
    # For simplicity and correctness, we do:
    # We will run this kernel with grid (NUM_EXPERTS,), and within each program, loop i in [0..NUM_EXPERTS-1].
    # That loop is allowed in Triton JIT for Python-side loops with runtime bounds.
    for i in range(NUM_EXPERTS):
        count_i = tl.load(counts_ptr + i)  # int32
        acc += count_i
        tl.store(offsets_ptr + i, acc)


@triton.jit
def stable_argsort_kernel(flat_ptr, sorted_ptr, offsets_ptr, M: tl.int32, NUM_EXPERTS: tl.int32):
    # Each program processes a chunk of indices. We'll implement per-index rank computation
    # and store into sorted_ptr.
    pid = tl.program_id(0)
    BLOCK = 1024
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < M
    vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # flat values

    # For each lane j in this block, compute rank = base_excl + tie_count
    # base_excl = offsets_ptr[vals[j]-1] if vals[j] > 0 else 0
    # tie_count = number of t < j with same key and vals[t] < vals[j]
    # We implement tie_count via scanning within the block.
    for j in tl.static_range(BLOCK):
        j_valid = mask[j]
        if j_valid:
            key_j = vals[j]
            base_excl = 0
            # base_excl: inclusive prefix sum up to key_j - 1
            if key_j > 0:
                base_excl = tl.load(offsets_ptr + (key_j - 1))
            # tie_count: count of earlier indices t < j with same key and vals[t] < vals[j]
            tie_count = 0
            # Scan only within this block's indices that come before j
            for t in tl.static_range(BLOCK):
                t_valid = mask[t]
                if t_valid and t < j:
                    key_t = tl.load(flat_ptr + t)
                    val_t = vals[t]
                    # Only count earlier t within this block with same key and smaller value
                    if key_t == key_j and val_t < key_j:
                        tie_count += 1
            rank = base_excl + tie_count
            tl.store(sorted_ptr + idx[j], rank)


# Optional: a tiny Triton kernel to compute sum of counts (if we ever need it without torch)
@triton.jit
def sum_counts_kernel(counts_ptr, total_ptr, NUM_EXPERTS: tl.int32):
    acc = 0
    for i in range(NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
    tl.store(total_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and dtype, flatten
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        M = flat.numel()
        device = flat.device
        NUM_EXPERTS = 256

        # 1) Count per expert using Triton
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        grid = (NUM_EXPERTS,)
        counts_by_exp_kernel[grid](flat, counts, M, NUM_EXPERTS)

        # 2) Inclusive prefix sums of counts using Triton (serial loop per output)
        # Note: offsets_incl is length NUM_EXPERTS
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        cumsum_inclusive_kernel[grid](counts, offsets_incl, NUM_EXPERTS)

        # 3) Compute expert_offsets: inclusive prefix sums and add 1 to the end
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        # Copy first NUM_EXPERTS
        expert_offsets[:NUM_EXPERTS] = offsets_incl
        # Compute total_count using torch.sum for simplicity (but if we must avoid torch, replace with sum_counts_kernel)
        total_count = int(counts.sum().item())
        expert_offsets[NUM_EXPERTS] = total_count + 1

        # 4) Stable argsort: produce sorted_token_indices (permutation of [0..M-1]) using Triton
        # Allocate output
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        # Launch stable argsort kernel over chunks
        BLOCK = 1024
        grid_argsort = (triton.cdiv(M, BLOCK),)
        stable_argsort_kernel[grid_argsort](flat, sorted_token_indices, offsets_incl, M, NUM_EXPERTS)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
