import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of values in flat_ptr (int32), one pass with atomic adds.
    flat_ptr: *int32, shape [N]
    counts_ptr: *int32, shape [num_experts]
    N: int, total number of elements
    BLOCK: int, number of elements processed per program
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        id_val = ids[i]
        valid = mask[i]
        if valid:
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def compute_le_counts(counts_ptr, le_counts_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive counts for each expert: le_counts[k] = sum_{j=0..k} counts[j].
    counts_ptr: *int32, length num_experts
    le_counts_ptr: *int32, length num_experts
    num_experts: compile-time constant
    """
    running = 0
    for k in range(0, num_experts):
        c = tl.load(counts_ptr + k)
        running += c
        tl.store(le_counts_ptr + k, running)


@triton.jit
def compute_lt_counts(le_counts_ptr, counts_ptr, lt_counts_ptr, num_experts: tl.constexpr):
    """
    Compute lt_counts[k] = le_counts[k] - counts[k] (number of elements < k).
    Used for stable tie-breaking: element equal to k goes at position number of elements strictly less than k.
    """
    for k in range(0, num_experts):
        le_k = tl.load(le_counts_ptr + k)
        c_k = tl.load(counts_ptr + k)
        tl.store(lt_counts_ptr + k, le_k - c_k)


@triton.jit
def compute_out_pos(flat_ptr, output_ptr, le_counts_ptr, lt_counts_ptr, N, BLOCK: tl.constexpr):
    """
    Triton kernel that computes the stable argsort permutation of original indices based on flat values.
    For each element i in flat:
      id = flat[i]
      pos = le_counts[id] - (counts[id] > 1 ? 1 : 0)  # stable: tie goes to earlier index
      write i (as int32) into output[pos]
    This kernel launches as a single-program kernel and scans all elements in chunks of BLOCK.
    """
    for base in range(0, N, BLOCK):
        offsets = base + tl.arange(0, BLOCK)
        mask = offsets < N
        pos_vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
        for i in range(BLOCK):
            id_val = pos_vals[i]
            valid = mask[i]
            if valid:
                le_id = tl.load(le_counts_ptr + id_val)
                lt_id = tl.load(lt_counts_ptr + id_val)
                cnt_id = le_id - lt_id  # number of occurrences of id_val
                tie_adj = 1 if cnt_id > 1 else 0
                pos = le_id - tie_adj
                # Store original index (offsets[i]) at sorted position pos
                tl.store(output_ptr + pos, offsets[i])


@triton.jit
def prefix_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length num_experts) into offsets_ptr (length num_experts).
    Host sets offsets_ptr[0] = 0. This kernel assumes we process the entire array; BLOCK must be >= num_experts.
    Given num_experts=256, set BLOCK=256.
    """
    running = 0
    for i in range(0, num_experts):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor for Triton
        assert topk_idx.is_cuda, "topk_idx must be a CUDA tensor for Triton kernels."
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32."

        # Flatten to 1D and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Fixed num_experts (as in original code)
        num_experts = 256

        # 1) Histogram via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST)

        # 2) Compute le_counts (inclusive) via Triton
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_le_counts[(1,)](counts, le_counts, num_experts)

        # 3) Compute lt_counts (exclusive less-than) via Triton
        lt_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_lt_counts[(1,)](le_counts, counts, lt_counts, num_experts)

        # 4) Compute stable argsort permutation via Triton and write output (launch required kernel)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        BLOCK_ARGSORT = 1024
        compute_out_pos[(1,)](flat, sorted_token_indices, le_counts, lt_counts, N, BLOCK_ARGSORT)

        # 5) Compute expert_offsets via Triton prefix scan
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0  # host sets zeroth offset
        BLOCK_EXPERTS = 256  # must be >= num_experts
        prefix_scan_kernel[(1,)](counts, expert_offsets, num_experts, BLOCK_EXPERTS)

        # Return results
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
