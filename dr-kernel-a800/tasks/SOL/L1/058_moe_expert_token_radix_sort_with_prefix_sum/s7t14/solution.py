import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened indices (int32).
# For each element in original_flat, atomically increment counts[value].
@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(original_ptr + offsets, mask=mask, other=0).to(tl.int32)
    # Only values in [0, NUM_VALUES-1] are meaningful; others will leave counts unchanged.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: inclusive prefix sum over counts array of length NUM_VALUES.
# prefix[i] = sum_{x<=i} counts[x] for i in [0..NUM_VALUES-1].
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    total = 0
    for i in range(NUM_VALUES):
        total += tl.load(counts_ptr + i)
        prefix_ptr[i] = total


# Triton kernel: stable permutation of original_flat -> sorted_token_indices.
# sorted_token_indices[i] = i, placed at position determined by value and local rank.
# For value v: number_of_less = sum_{x<v} counts[x]; then for each equal element we assign
# its position as number_of_less + (rank among equals computed by scanning original_flat).
@triton.jit
def stable_permutation_kernel(original_ptr, counts_ptr, prefix_ptr, sorted_indices_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # We iterate values v and compute number_of_less for each. Then we scan original_flat in chunks
    # and assign each index i with value v to position number_of_less + local_rank.
    # Note: Since values are in [0..NUM_VALUES-1], we can compute number_of_less as prefix[v-1] if v > 0; else 0.
    for v in range(NUM_VALUES):
        # number_of_less = sum_{x<v} counts[x]
        if v == 0:
            number_of_less = 0
        else:
            number_of_less = tl.load(prefix_ptr + (v - 1))

        # Now place all indices i such that original[i] == v.
        # We process in chunks and compute local_eq_before for each chunk.
        for base in range(0, M, BLOCK):
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(original_ptr + offsets, mask=mask, other=0).to(tl.int32)
            # Count how many positions in this chunk have value v.
            eq_count = 0
            for i in range(BLOCK):
                if mask[i]:
                    vi = vals[i]
                    if vi == v:
                        eq_count += 1

            # Compute local ranks for each element in this chunk:
            # rank is the count of equals that appear before it (based on original offsets).
            for i in range(BLOCK):
                if mask[i]:
                    vi = vals[i]
                    # only process equals
                    if vi == v:
                        # count how many equals before this position in original offsets
                        local_eq_before = 0
                        for j in range(BLOCK):
                            if mask[j]:
                                vj = vals[j]
                                if vj == v and offsets[j] < offsets[i]:
                                    local_eq_before += 1
                        pos = number_of_less + local_eq_before
                        # sorted_indices_ptr is 1D output buffer for positions, but we actually need to store
                        # the original index i at position pos. We cannot directly scatter; instead we keep
                        # a separate output buffer sorted_indices_out_ptr (int32) and write i at pos.
                        tl.store(sorted_indices_ptr + pos, offsets[i])


# Triton helper: fill offsets[0..NUM_VALUES] with inclusive prefix sums; return offsets[0..NUM_VALUES].
@triton.jit
def assemble_offsets_kernel(counts_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    prefix = tl.zeros(NUM_VALUES, dtype=tl.int32)
    total = 0
    for i in range(NUM_VALUES):
        total += tl.load(counts_ptr + i)
        prefix[i] = total
    # Write inclusive prefix sums into offsets_ptr[1..NUM_VALUES]
    # offsets_ptr[0] will be set by host to 0.
    for i in range(NUM_VALUES):
        offsets_ptr[i + 1] = prefix[i]


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.NUM_VALUES = 256  # matches num_experts_per_tok in provided workloads

    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs:
          topk_idx: (batch_size, seq_len, num_experts_per_tok) int32 tensor on device.
        Outputs:
          sorted_token_indices: (M,) int32, permutation that would sort topk_idx flattened stably.
          expert_offsets: (num_experts_per_tok + 1,) int32, cumulative histogram inclusive.
        """
        assert topk_idx.dtype == torch.int32
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"

        # Flatten original and keep a copy for Triton kernels
        original_flat = topk_idx.reshape(-1).contiguous()  # 1D int32
        M = original_flat.numel()
        device = original_flat.device

        # 1) Histogram of values in [0..NUM_VALUES-1]
        counts = torch.zeros(self.NUM_VALUES, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](original_flat, counts, M, self.NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sums to compute offsets (length NUM_VALUES+1)
        # We'll use Triton to compute prefix and assemble offsets array.
        prefix = torch.empty(self.NUM_VALUES, dtype=torch.int32, device=device)
        # Triton kernel to compute prefix
        prefix_sum_kernel[(1,)](counts, prefix, self.NUM_VALUES)
        offsets = torch.empty(self.NUM_VALUES + 1, dtype=torch.int32, device=device)
        # Assemble offsets: offsets[0] = 0, offsets[i+1] = prefix[i]
        offsets[0] = 0
        # We can fill remaining using simple torch addition, or call a tiny Triton kernel:
        # offsets[1:] = prefix
        offsets[1:] = prefix

        # 3) Stable permutation: sorted_token_indices is the permutation such that
        #    sorted_token_indices[i] is the position of original_flat[i] in stable order.
        #    We use Triton kernel to compute this.
        sorted_indices = torch.empty(M, dtype=torch.int32, device=device)

        # Ensure counts and prefix are available to Triton; sorted_indices is the output buffer.
        grid_perm = (1,)  # single program; loops inside kernel handle all elements
        stable_permutation_kernel[grid_perm](original_flat, counts, prefix, sorted_indices, M, self.NUM_VALUES, BLOCK)

        return sorted_indices, offsets


def run(*args):
    return ModelNew()(*args)
