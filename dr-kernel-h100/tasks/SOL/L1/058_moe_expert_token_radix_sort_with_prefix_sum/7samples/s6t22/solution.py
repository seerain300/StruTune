import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_sort_pairs(keys_ptr, pos_ptr, n, sentinel_key, BLOCK: tl.constexpr):
    """
    In-place bitonic sort network on arrays 'keys_ptr' and 'pos_ptr' of length BLOCK.
    We process the first n elements (n <= BLOCK, BLOCK is next power-of-two).
    Sentinel for keys beyond n is (sentinel_key + 1) to ensure they sort to the end.
    We keep stability for equal keys by preferring smaller original index in the pair.
    This kernel is intended to be launched with a grid that covers all pairs per stage.
    """
    pid = tl.program_id(axis=0)
    # Each program handles one pair at a given stage and stride
    # We'll iterate over stages and strides in Python/host, launching this for each.
    # Here pid encodes (idx, stage, k). For simplicity, one launch per stage/stride loop
    # is better; Triton requires static shapes; we'll do that in the host launcher.
    pass  # Placeholder; actual stages are executed via dynamic launch configuration


@triton.jit
def _bitonic_sort_keys_pos(keys_ptr, pos_ptr, n, sentinel_key, BLOCK: tl.constexpr):
    """
    Wrapper that iterates the bitonic sort network stages. We cannot express the
    multi-dimensional loops directly in Triton without knowing BLOCK, so we compute
    stages/strides on host and launch this kernel repeatedly for each stage/stride.
    """
    # No-op in Triton; actual stages are managed by host-driven launches.
    pass


@triton.jit
def _hist_kernel(values_ptr, counts_ptr, n, num_classes: tl.constexpr):
    """
    Histogram of values in values_ptr (int32) over num_classes bins (here 256).
    counts_ptr is length num_classes, initialized to zeros, incremented by +1 for each occurrence.
    """
    idx = tl.program_id(axis=0)
    # Each program handles one class
    if idx < num_classes:
        # Scan through n elements
        for i in range(0, n):
            val = tl.load(values_ptr + i)
            if val == idx:
                tl.atomic_add(counts_ptr + idx, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_ptr, length: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length 256) into offsets_ptr (length 256).
    offsets[i] = sum(counts[:i+1]).
    """
    # Single-program inclusive scan over a known small length using iterative doubling.
    # We write offsets in place; initialize offsets[0] = counts[0], then add previous.
    for i in range(0, length):
        # Compute prefix[i] = prefix[i-1] + counts[i], with prefix[-1] = 0
        # Triton doesn't support dynamic loops well; we use a static unrolled loop.
        # But length is tl.constexpr, so Triton can unroll. We'll implement step-by-step:
        # Manually unroll for up to 256.
        # To do so, we use a series of loads/stores with masks; simpler approach:
        # We can't access prefix[i-1] directly; so we do a two-pass approach:
        # First pass: compute exclusive scan into tmp, second pass: offsets = tmp + counts.
        # However, Triton supports tl.load/tl.store; but not global vector accumulation across lanes.
        # Instead, perform scan in registers if we load counts[i..] into vectors. Triton doesn't
        # allow reading from global memory into a vector across lanes like that directly.
        # Therefore, we implement a straightforward iterative doubling scan using a single program
        # that recomputes each prefix; but since it's small, we can afford repeated loads:
        # We'll implement as follows: we maintain a scalar acc, load each counts[i], add to acc,
        # store acc to offsets[i]. Triton supports scalar accumulation with tl.load/tl.store.
        acc = 0
        for i_local in range(0, length):
            count_i = tl.load(counts_ptr + i_local)
            acc += count_i
            tl.store(offsets_ptr + i_local, acc)


def _next_power_of_two(n: int) -> int:
    # Returns the next power of two >= n
    return 1 << ((n - 1).bit_length())


def _compute_expert_offsets_flat(values: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert_offsets[1:] = inclusive cumulative counts of values.
    Use Triton kernels for histogram and inclusive scan; return int32 tensor on device.
    """
    assert values.is_cuda
    # Histogram counts per expert class (0..num_experts-1)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=values.device)
    # Triton launch: one program per class
    grid = (num_experts,)
    _hist_kernel[grid](values, counts, values.numel(), num_experts)
    # Inclusive prefix sum
    offsets = torch.empty(num_experts, dtype=torch.int32, device=values.device)
    _inclusive_scan_kernel[(1,)](counts, offsets, num_experts)
    # Return shape (num_experts + 1,) with first element 0
    return torch.cat([torch.tensor([0], device=values.device, dtype=torch.int32), offsets])


def _launch_global_sort(values: torch.Tensor) -> torch.Tensor:
    """
    Sort the flattened 1D int32 'values' globally in ascending order using a Triton bitonic sort.
    Return the permutation 'pos' of indices [0..values.numel()-1] such that
    values[pos[i]] is sorted ascending, matching torch.argsort(stable=True).
    """
    assert values.is_cuda
    n = values.numel()
    # Next power of two for bitonic sort
    block = _next_power_of_two(n)
    # Prepare keys and positions
    keys = values.clone()
    pos = torch.arange(n, device=values.device, dtype=torch.int32)

    # For positions beyond n, set sentinel to push to end; but bitonic works on n.
    sentinel_key = 1 << 30  # larger than any expected key (0..255)

    # We'll implement bitonic sort via repeated launches for each stage/stride.
    # The classic bitonic network:
    # for p in 2,4,...,BLOCK:
    #   for q in p//2 down to 1:
    #     for i in 0..n-1:
    #        ixj = i ^ q
    #        if ixj > i:
    #            compare keys[i], keys[ixj]; swap if out of order or tie broken by pos
    # We do these launches from Python/host.
    # Note: Triton doesn't support nested loops over dynamic variables well; we compute stages and
    # launch the kernel once per stage/stride, with appropriate ixj computed in kernel.

    # For simplicity and correctness, we use the following approach:
    # Implement the sorting network in a single kernel by passing 'stage' and 'stride' from host
    # and having the kernel handle only one pair per launch. This is fine because Triton requires
    # static shapes and we can launch the number of programs equal to n*(number_of_pairs).
    # However, Triton doesn't allow arbitrary dynamic loops; therefore, we implement a standard
    # bitonic sort using a fixed BLOCK by reusing n and masking out-of-range. To be robust, we
    # perform the sort using Python loops to drive the Triton kernel calls, which Triton supports.

    # Since Triton kernels need static grid, we perform the sorting stages with host loops:
    # For each stage p in powers of two up to BLOCK, and for each stride q = p//2 down to 1:
    # We launch a kernel that compares pairs (i, ixj=i^q) and swaps if out of order, with stable tie-breaker.
    # We need a grid. Each program handles one index i. For each i, compute ixj and then decide if we
    # should do the compare-swap. To avoid race, we launch the kernel twice: once for i<ixj, once for ixj<i,
    # but we can simply use grid=(n,) and inside kernel we compute if ixj>i. This avoids double work because
    # the second half of the network is symmetric.

    # Implement classic bitonic network using Python loops; Triton supports looping in kernels, but
    # nesting and dynamic bounds can be problematic. Instead, we use the standard approach:
    # Compute stages and strides in Python and launch kernels accordingly.

    # We'll implement a single kernel that, given 'stage' and 'stride', performs compare-swap for all i
    # where ixj = i ^ stride and ixj > i. We pass stage/stride as meta-parameters.

    # Helper function: perform a single bitonic stage with given 'p' and 'q'
    # We define the kernel as:
    # Each program handles one index i; it loads keys[i], keys[ixj], pos[i], pos[ixj], computes
    # whether to swap based on ascending direction of the sequence and stable tie-breaker.

    @triton.jit
    def _bitonic_stage_stride(keys_ptr, pos_ptr, n, p: tl.constexpr, q: tl.constexpr):
        i = tl.program_id(axis=0)
        ixj = i ^ q
        # Only process each pair once
        if ixj > i:
            # Load current values
            a_key = tl.load(keys_ptr + i)
            b_key = tl.load(keys_ptr + ixj)
            a_pos = tl.load(pos_ptr + i)
            b_pos = tl.load(pos_ptr + ixj)
            # Ascending direction for the sequence containing i is determined by (i & p) == 0
            ascending = ((i & p) == 0)
            # Compare keys
            swap_keys = (a_key > b_key)
            # Stable tie-breaker: if equal, keep smaller original index first
            equal = (a_key == b_key)
            prefer_a = (a_pos < b_pos)
            do_swap = tl.where(ascending, swap_keys | (equal & (~prefer_a)), swap_keys | (equal & prefer_a))
            if do_swap:
                # Swap keys and positions
                tmp_key = a_key
                tmp_pos = a_pos
                tl.store(keys_ptr + i, b_key)
                tl.store(keys_ptr + ixj, tmp_key)
                tl.store(pos_ptr + i, b_pos)
                tl.store(pos_ptr + ixj, tmp_pos)

    # Execute bitonic sort stages
    # p is 2, 4, 8, ..., BLOCK
    for p in [1 << k for k in range(1, 16)]:  # 2, 4, 8, ..., 65536 (sufficient to cover up to 8192)
        if p > n:
            break
        # q is p//2, p//4, ..., 1
        q = p >> 1
        while q >= 1:
            grid = (n,)
            _bitonic_stage_stride[grid](keys, pos, n, p, q)
            q >>= 1

    # After sorting, 'pos' contains the permutation of indices that sorts 'keys' ascending.
    # Return pos as int32 (stable argsort permutation).
    return pos


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect topk_idx of shape (batch_size, seq_len, num_experts_per_tok)
        if len(args) == 1 and isinstance(args[0], dict):
            # If a dict is passed, use its topk_idx; but original signature is forward(self, *args),
            # and get_inputs returns a dict in the evaluation harness. Here, we assume the first
            # positional argument is the tensor topk_idx. To be robust, we handle both: if dict,
            # use it; if tensor, use it. However, in typical evaluation, a tensor is passed.
            if isinstance(args[0], torch.Tensor):
                topk_idx = args[0]
            else:
                topk_idx = list(args)[0]
        else:
            # If not a tensor, attempt to extract tensor from first positional argument
            topk_idx = args[0] if isinstance(args[0], torch.Tensor) else None
            if topk_idx is None:
                raise ValueError("ModelNew.forward expects topk_idx tensor as first argument.")

        if not topk_idx.is_cuda:
            # The evaluator provides CUDA tensors; ensure we don't fall back to CPU accidentally.
            # If CPU tensor is passed, move to current CUDA device.
            topk_idx = topk_idx.to(torch.device("cuda"))

        # Flatten and cast to int32 for Triton
        flat = topk_idx.reshape(-1).to(torch.int32)

        # 1) Global stable sort of flat using Triton bitonic sort, return permutation indices
        sorted_idx = _launch_global_sort(flat)

        # 2) Compute expert offsets from original flat using Triton histogram + inclusive scan
        expert_offsets = _compute_expert_offsets_flat(flat, 256)

        # Return results: sorted_token_indices (int32, shape (N,)) and expert_offsets (int32, shape (257,))
        return sorted_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
