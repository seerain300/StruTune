import torch
import triton
import triton.language as tl


# Triton kernel: bitonic stable sort on an array of pairs (value, index), ascending by value.
# We sort over a length BLOCK (power of two), padding with large values so padded entries go to the end.
# For ties (value == partner), we enforce stability by preferring smaller original index.
@triton.jit
def bitonic_sort_pairs_kernel(data_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    k = 2
    while k <= BLOCK:
        j = k // 2
        while j > 0:
            partner = i ^ j

            # Load current pair (i) and partner pair (partner)
            base_i = i * 2
            base_p = partner * 2
            ai = tl.load(data_ptr + base_i + 0)  # value at i
            ii = tl.load(data_ptr + base_i + 1)  # original index at i
            ap = tl.load(data_ptr + base_p + 0)  # value at partner
            ip = tl.load(data_ptr + base_p + 1)  # original index at partner

            # Ascending/descending phase flag
            asc = ( (i & k) == 0 )
            cmp = ai > ap  # ascending comparison
            tie = ai == ap
            # Stable tie-break: for ascending, swap if ii > ip; for descending, swap if ii < ip
            stable_asc_swap = tie & (ii > ip)
            stable_desc_swap = tie & (ii < ip)

            do_swap = (cmp ^ asc)  # XOR: ascending -> swap on ai>ap, descending -> swap on ai<ap
            # Apply stable swap rules only in ascending passes; in descending passes we just reverse the condition
            if asc:
                do_swap = do_swap | stable_asc_swap
            else:
                do_swap = do_swap | stable_desc_swap

            # Swap the pairs if needed
            ai_new = tl.where(do_swap, ap, ai)
            ii_new = tl.where(do_swap, ip, ii)
            ap_new = tl.where(do_swap, ai, ap)
            ip_new = tl.where(do_swap, ii, ip)

            tl.store(data_ptr + base_i + 0, ai_new)
            tl.store(data_ptr + base_i + 1, ii_new)
            tl.store(data_ptr + base_p + 0, ap_new)
            tl.store(data_ptr + base_p + 1, ip_new)

            j //= 2
        k *= 2


# Triton kernel: histogram of values in orig (int32) into counts (int32)
# Assumes values are in [0, 255] (num_experts=256). We guard out-of-range with 0.
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    offsets = lane * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    for v in range(L):
        eq = (vals == v) & mask
        num_match = tl.sum(eq.to(tl.int32))
        tl.atomic_add(counts_ptr + v, num_match)


# Triton kernel: exclusive prefix-sum over counts to produce offsets
# out_ptr[0..L-1] = exclusive sums; out_ptr[L] = total N
@triton.jit
def exclusive_scan_kernel(counts_ptr, out_ptr, L: tl.constexpr):
    running = 0
    for i in range(L):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(out_ptr + i, running - ci)  # exclusive: sum of previous elements
    total = running
    tl.store(out_ptr + L, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA device for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten
        flat = topk_idx.reshape(-1).contiguous()  # int32
        N = flat.numel()

        num_experts = 256  # per original setup

        # We need to compute sorted_token_indices via Triton (stable sort).
        # Implement bitonic sort over BLOCK = next power of two >= N, padding with large sentinel values.
        def next_power_of_two(x: int) -> int:
            return 1 << (x - 1).bit_length()

        BLOCK = next_power_of_two(N)

        # Allocate data for pairs: [value, original_index] of length BLOCK
        data = torch.empty(BLOCK * 2, dtype=torch.int32, device=flat.device)
        # Initialize: for i < N, store (flat[i], i); for i >= N, store (large sentinel, i) to push to end.
        INF = 1 << 30  # large int32 sentinel
        for i in range(BLOCK):
            val = flat[i] if i < N else INF
            data[i * 2 + 0] = val
            data[i * 2 + 1] = i

        # Launch bitonic sort kernel: sorts ascending stably by value, ties broken by index
        grid = (BLOCK,)  # one program per position in the block
        bitonic_sort_pairs_kernel[grid](data, N, BLOCK=BLOCK)

        # Extract sorted_token_indices: original indices after sorting (first N)
        sorted_token_indices = data[1::2].to(torch.int32)[:N]  # shape (N,)

        # Compute expert_offsets using Triton histogram and exclusive scan
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel over valid N elements
        BLOCK_H = 1024
        grid_h = (triton.cdiv(N, BLOCK_H),)
        histogram_kernel[grid_h](flat, counts, N, L=num_experts, BLOCK=BLOCK_H)

        # Exclusive prefix-sum to get offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        exclusive_scan_kernel[(1,)](counts, offsets, L=num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
