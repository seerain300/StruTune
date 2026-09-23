import torch
import triton
import triton.language as tl


@triton.jit
def count_values_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # One program per index i
    i = tl.program_id(0)
    # Process BLOCK elements per program; i may exceed N, but we guard with mask.
    # Since grid=(N,), each program handles a single element. We keep it simple.
    if i < N:
        v = tl.load(flat_ptr + i)
        # Increment count for v (int32)
        tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def exclusive_scan_const_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Compute exclusive prefix sums for counts_ptr[0..L-1] -> offsets_ptr[0..L-1]
    # Use sequential scan within the kernel. L is constexpr for unrolling.
    # We need to loop from 0 to L-1; Triton allows static loops when L is constexpr.
    # offsets_ptr is length L (values), we can extend to L+1 on host and set last to N after.
    acc = 0
    for j in range(0, L):
        val = tl.load(counts_ptr + j)
        acc += val
        tl.store(offsets_ptr + j, acc - val)


@triton.jit
def stable_rank_kernel(flat_ptr, counts_ptr, offsets_ptr, ranks_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    # One program per index i
    i = tl.program_id(0)
    if i < N:
        v = tl.load(flat_ptr + i)
        # Compute stable local rank r: number of j < i with flat[j] <= v
        # We iterate j in chunks of BLOCK up to i-1, sum a boolean mask, and use atomic_add to accumulate into ranks_ptr[i].
        r = tl.zeros((), dtype=tl.int32)
        # Loop over j=0..i-1 in blocks of size BLOCK
        for start in range(0, i, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            mask = idx < i
            # Load flat[idx] with mask; other elements won't affect sum
            vj = tl.load(flat_ptr + idx, mask=mask, other=0)
            cmp = (vj <= v) & mask
            # Sum boolean mask to int32
            r += tl.sum(cmp.to(tl.int32))
        # Atomically add r into ranks_ptr[i]
        tl.atomic_add(ranks_ptr + i, r)


@triton.jit
def place_stable_indices_kernel(ranks_ptr, offsets_ptr, sorted_ptr, flat_ptr, N, L: tl.constexpr):
    # One program per index i
    i = tl.program_id(0)
    if i < N:
        v = tl.load(flat_ptr + i)
        r = tl.load(ranks_ptr + i)
        pos = tl.load(offsets_ptr + v) + r
        tl.store(sorted_ptr + pos, i)


@triton.jit
def histogram_original_experts_kernel(orig_ptr, counts_exp_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    # L = num_experts = 256. We count occurrences of each expert id 0..L-1 in orig_ptr[0..N-1].
    for e in range(0, L):
        # Each program handles one expert e and loops over orig
        total = tl.zeros((), dtype=tl.int32)
        for start in range(0, N, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            mask = idx < N
            val = tl.load(orig_ptr + idx, mask=mask, other=0)
            # For masked elements, set other to a value outside [0..L-1] so it won't match; but since orig_ptr is int32, we must ensure no false match.
            eq = (val == e) & mask
            total += tl.sum(eq.to(tl.int32))
        # Store total count for expert e
        tl.store(counts_exp_ptr + e, total)


@triton.jit
def exclusive_scan_const_experts_kernel(counts_exp_ptr, offsets_exp_ptr, L: tl.constexpr):
    # Compute exclusive prefix sums for counts_exp_ptr[0..L-1] -> offsets_exp_ptr[0..L-1]
    acc = 0
    for j in range(0, L):
        val = tl.load(counts_exp_ptr + j)
        acc += val
        tl.store(offsets_exp_ptr + j, acc - val)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Extract flattened original topk_idx and the values (flat) we will sort
        orig = topk_idx.view(-1)
        N = orig.numel()
        device = orig.device

        # Prepare outputs
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        # For values in [0..255], counts_vals has length 256
        counts_vals = torch.zeros(256, dtype=torch.int32, device=device)
        offsets_vals = torch.empty(256, dtype=torch.int32, device=device)

        # Launch counting of values
        grid = (N,)
        count_values_kernel[grid](orig, counts_vals, N, BLOCK=1024)
        # Exclusive scan for values to get offsets[v]
        exclusive_scan_const_kernel[(1,)](counts_vals, offsets_vals, L=256)

        # Stable ranks buffer (per index)
        ranks = torch.zeros(N, dtype=torch.int32, device=device)

        # Compute stable local ranks
        stable_rank_kernel[grid](orig, counts_vals, offsets_vals, ranks, N, L=256, BLOCK=1024)
        # Place indices at their stable positions
        place_stable_indices_kernel[grid](ranks, offsets_vals, sorted_token_indices, orig, N, L=256)

        # Expert offsets:
        # Counts per expert id 0..255 from original topk_idx (orig)
        counts_exp = torch.zeros(256, dtype=torch.int32, device=device)
        # We need to count occurrences of each expert id in orig. Since orig is int32, we just loop and compare.
        # Launch histogram kernel over N elements
        # Note: grid=(256,) means one program per expert id; inside each program, we loop over N in chunks of BLOCK=1024.
        grid_exp = (256,)
        histogram_original_experts_kernel[grid_exp](orig, counts_exp, N, L=256, BLOCK=1024)

        offsets_exp = torch.empty(257, dtype=torch.int32, device=device)  # we'll fill [0..255] via Triton, and set last to N
        exclusive_scan_const_experts_kernel[(1,)](counts_exp, offsets_exp, L=256)

        # Set last element to total N (host write for clarity; single scalar)
        offsets_exp[-1] = N

        return sorted_token_indices, offsets_exp


def run(*args):
    return ModelNew()(*args)
