import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.int32,
                              BLOCK_SIZE: tl.constexpr):
    """
    Build per-expert counts using atomic add. Assumes flat_ptr points to int32 values
    and counts_ptr is length num_experts (256).
    Grid size: (ceil_div(n_elements, BLOCK_SIZE),)
    Each program processes a chunk of BLOCK_SIZE elements.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # For out-of-range lanes, set val to -1 so they don't contribute
    vals = tl.where(mask, vals, -1)
    # Atomic add counts for valid lanes
    for i in range(BLOCK_SIZE):
        val = vals[i]
        # Only atomically add for valid lanes with val in [0, 255]
        valid = mask[i] & (val >= 0) & (val <= 255)
        if valid:
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr,
                                  num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr (length num_experts) into offsets_ptr
    (length num_experts+1). We do a simple sequential scan within the kernel:
    offsets[i] = sum(counts[:i]) for i=1..num_experts; offsets[0]=0.
    Grid size: (1,)
    """
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # Sequential inclusive scan
    for i in range(1, num_experts + 1):
        # sum of counts[:i] = previous offset[i-1] + counts[i-1]
        prev = tl.load(offsets_ptr + (i - 1))
        cur = tl.load(counts_ptr + (i - 1))
        tl.store(offsets_ptr + i, prev + cur)


@triton.jit
def _counting_sort_with_indices_kernel(
    flat_ptr, indices_ptr, offsets_ptr, out_indices_ptr,
    n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr
):
    """
    Perform counting sort and produce sorted_token_indices (permutation of 0..n_elements-1).
    We write the permutation into out_indices_ptr.
    Each program processes a chunk of BLOCK_SIZE elements:
    - Load values from flat_ptr.
    - Compute for each value its offset from offsets_ptr and place the corresponding original index
      from indices_ptr into out_indices at that position.
    We rely on counts being computed and offsets being inclusive such that for each expert e,
    offset[e] is the starting position for tokens with value e, and offset[e+1]-offset[e] is the
    number of tokens with value e. Since we place indices in increasing e order, this yields a
    stable sort by value, with original order preserved for equal values.
    Grid size: (ceil_div(n_elements, BLOCK_SIZE),)
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load original indices (0..n_elements-1)
    orig = tl.load(indices_ptr + offsets, mask=mask, other=0)  # int32

    # Load values (assume valid range 0..num_experts-1)
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32

    for i in range(BLOCK_SIZE):
        pos = offsets[i]
        valid = mask[i]
        if valid:
            # Place original index 'orig[i]' at position 'offsets[vals[i]]'
            # We must guard offsets_ptr access with valid val in [0, num_experts-1].
            # Compute position for this value
            dest = tl.load(offsets_ptr + vals[i])
            # Store original index at dest in out_indices
            tl.store(out_indices_ptr + dest, orig[i])


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT) int32 tensor on CUDA device.
        Returns:
          sorted_token_indices: (N,) int64 tensor (permutation of 0..N-1) that sorts topk_idx flattened.
          expert_offsets: (num_experts+1,) int32 tensor, inclusive prefix sums of counts.
        """
        # Ensure CUDA and contiguous
        assert topk_idx.is_cuda, "ModelNew requires a CUDA tensor"
        flat = topk_idx.reshape(-1).contiguous()

        # 1) Triton histogram of expert IDs
        n = flat.numel()
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        # Grid over chunks of BLOCK_SIZE
        BLOCK_SIZE = 1024
        grid_h = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid_h](flat, counts, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Triton inclusive prefix sum of counts -> offsets (length num_experts+1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # 3) Prepare original indices 0..n-1 and Triton sorting to produce permutation
        indices = torch.arange(n, dtype=torch.int32, device=flat.device)
        out_indices = torch.empty(n, dtype=torch.int32, device=flat.device)

        # Launch sorting kernel
        grid_sort = (triton.cdiv(n, BLOCK_SIZE),)
        _counting_sort_with_indices_kernel[grid_sort](
            flat, indices, offsets, out_indices, n, self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Return sorted_token_indices as int64 to match original, and expert_offsets as int32
        sorted_token_indices = out_indices.to(torch.int64)
        return sorted_token_indices, offsets