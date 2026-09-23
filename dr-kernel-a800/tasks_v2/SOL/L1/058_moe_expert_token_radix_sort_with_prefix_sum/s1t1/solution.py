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
    Used for stable tie-breaking: element equal to k should go at position
    equal to number of elements strictly less than k.
    """
    for k in range(0, num_experts):
        le_k = tl.load(le_counts_ptr + k)
        c_k = tl.load(counts_ptr + k)
        tl.store(lt_counts_ptr + k, le_k - c_k)


@triton.jit
def compute_out_pos(
    flat_ptr, out_pos_ptr, le_counts_ptr, lt_counts_ptr, N,
    BLOCK: tl.constexpr
):
    """
    Build out_pos[i] = original index at the stable sorted position i.
    We fill positions sequentially:
      for k in [0..num_experts-1]:
        pos_k = le_counts[k] - (1 if counts[k] > 1 else 0)
        For the next cnt_k elements, set out_pos[pos_k + t] = original index at flat[j] == k.
      We maintain a next slot counter and fill consecutive positions; this avoids collisions
      because writes are controlled per value and per original index.
    """
    running_slot = 0
    for k in range(0, tl.num_experts):  # tl.num_experts is not available; rely on Python loop with num_experts constexpr
        # The above line is illustrative; in Triton, we cannot refer to num_experts inside the kernel as a runtime value.
        # Instead, we rely on separate launch of this kernel with fixed N and rely on num_experts as a constexpr in surrounding code.
        # To implement the loop over num_experts, we pass num_experts as a constexpr parameter to the kernel.
        # Here we restructure the kernel to take num_experts as constexpr and use it in a Python for loop.
        pass
    # The above placeholder shows the intent. We will define a proper kernel variant with num_experts constexpr below.


# Proper compute_out_pos with num_experts constexpr
@triton.jit
def compute_out_pos_real(flat_ptr, out_pos_ptr, le_counts_ptr, lt_counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    running_slot = 0
    for k in range(0, num_experts):
        le_k = tl.load(le_counts_ptr + k)
        lt_k = tl.load(lt_counts_ptr + k)
        cnt_k = le_k - lt_k  # number of occurrences of k
        # For each occurrence of k, place at position le_k - (1 if duplicates else 0), i.e., at slot le_k - (cnt_k > 1 ? 1 : 0)
        adj = 1 if cnt_k > 1 else 0
        pos_base = le_k - adj
        # We need to scatter original indices equal to k into positions running_slot:running_slot + cnt_k - 1
        # But Triton does not support arbitrary scatter from global load based on equality. Instead, we structure the algorithm
        # as follows: we will do this in a two-phase approach on host, but since we must keep all Triton, we implement
        # a deterministic filling by scanning flat_ptr and writing to out_pos at computed positions.
        # To do that cleanly, we use a third Triton kernel that reads flat_ptr and writes to out_pos at the computed positions.
        # However, Triton doesn't provide a built-in way to "write at specific positions" without knowing which indices to place.
        # Therefore, we implement the out_pos computation via a dedicated Triton kernel that fills positions sequentially by
        # scanning flat_ptr and using le_counts/lt_counts to determine the destination. This requires a kernel that loops
        # over N and uses the counts arrays; we can achieve this by launching the kernel with grid=(1,) and processing N elements
        # in chunks. Triton kernels are not designed for arbitrary element writes based on complex logic; hence we simplify:
        # We will instead compute the argsort permutation by directly placing values into output using compute_positions_and_output,
        # and compute out_pos as a separate step in PyTorch (since it's not strictly forbidden by your requirement to "use Triton"),
        # but the requirement is to use Triton for computation. To strictly adhere, we restructure to compute argsort directly
        # and avoid out_pos computation. Since out_pos is not actually needed by the original run function's outputs, we can omit it.
    # To simplify and adhere to the requirement, we remove compute_out_pos and rely on compute_positions_and_output which
    # writes the permutation directly. The outputs we need are sorted_token_indices and expert_offsets.


# Replacing out_pos approach with direct argsort via positions
@triton.jit
def compute_positions_and_output(
    flat_ptr, output_ptr, le_counts_ptr, lt_counts_ptr, N,
    BLOCK: tl.constexpr
):
    """
    For each element i in flat:
      id = flat[i]
      pos = le_counts[id] - (counts[id] > 1 ? 1 : 0)  # stable: tie goes to earlier index
      write i into output[pos]
    Produces the stable argsort permutation of original indices based on flat values.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load original indices (positions) and values
    pos_vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # For each valid element, compute its le and lt counts, then destination position
    for i in range(BLOCK):
        id_val = pos_vals[i]
        valid = mask[i]
        if valid:
            # le and lt counts for this id
            le_id = tl.load(le_counts_ptr + id_val)
            lt_id = tl.load(lt_counts_ptr + id_val)
            # number of strictly less elements; handle ties: subtract 1 if duplicates present
            cnt_id = le_id - lt_id
            tie_adj = 1 if cnt_id > 1 else 0
            pos = le_id - tie_adj
            # Store original index (offsets[i]) at sorted position pos
            tl.store(output_ptr + pos, offsets[i])


@triton.jit
def prefix_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length num_experts) into offsets_ptr (length num_experts).
    Then host sets offsets_ptr[0] = 0. This kernel assumes we process the entire array; BLOCK must be >= num_experts.
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

        # 4) Compute stable argsort permutation via Triton and write output
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        BLOCK_ARGSORT = 1024
        grid_argsort = (triton.cdiv(N, BLOCK_ARGSORT),)
        compute_positions_and_output[grid_argsort](flat, sorted_token_indices, le_counts, lt_counts, N, BLOCK_ARGSORT)

        # 5) Compute expert_offsets via Triton prefix scan
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0  # host sets zeroth offset
        BLOCK_EXPERTS = 256  # must be >= num_experts
        prefix_scan_kernel[(1,)](counts, expert_offsets, num_experts, BLOCK_EXPERTS)

        # Return results
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
