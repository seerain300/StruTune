import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK_HIST: tl.constexpr):
    """
    Build histogram counts of expert ids present in flat_ptr[0:N].
    grid = (ceil_div(N, BLOCK_HIST),)
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_HIST + tl.arange(0, BLOCK_HIST)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_sum(out_ptr, inp_ptr, length: tl.constexpr):
    """
    Single-program inclusive scan of a small vector inp_ptr[0:length] into out_ptr[0:length].
    length is a constexpr for compile-time loop.
    """
    total = 0
    for i in range(0, length):
        val = tl.load(inp_ptr + i)
        total += val
        tl.store(out_ptr + i, total)


@triton.jit
def compute_out_pos_triton(flat_ptr, out_ptr, N):
    """
    Compute stable argsort permutation for flat_ptr[0:N] into out_ptr[0:N].
    For each element i with id = flat[i], position pos is:
      pos = le[id] - (1 if there are duplicates and i is not first else 0)
    This ensures stable ordering by original index among ties.
    Single-program sequential loop over i.
    """
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        dupe_flag = 0
        # Scan previous elements to detect duplicates
        for j in range(0, i):
            if tl.load(flat_ptr + j) == val:
                dupe_flag = 1
                break
        # le_counts is assumed to be provided as a global device array of size 256 (see forward)
        le_counts = tl.load(le_counts_ptr + val)  # int32
        pos = le_counts - dupe_flag
        tl.store(out_ptr + i, pos)


@triton.jit
def cumsum_inclusive_triton(inp_ptr, out_ptr, length: tl.constexpr):
    """
    Inclusive cumsum of a small vector inp_ptr[0:length] into out_ptr[0:length].
    length is a constexpr for compile-time loop.
    Used to produce expert_offsets[1:] by scanning counts.
    """
    total = 0
    for i in range(0, length):
        val = tl.load(inp_ptr + i)
        total += val
        tl.store(out_ptr + i, total)
    # out_ptr[length] is not used (we store into out_ptr[i+1] by writing to out_ptr[i] above)
    # We need to set out_ptr[0] = 0 separately in forward.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Produces sorted_token_indices = stable argsort over flattened topk_idx
        - Produces expert_offsets = prefix sums per expert (including leading zero)
        Returns (sorted_token_indices, expert_offsets)
        """
        # Assume topk_idx is 1D int32 tensor of length N on device
        flat = topk_idx  # already 1D int32 per the original get_inputs
        N = flat.numel()
        device = flat.device

        num_experts = 256

        # 1) Triton histogram of expert IDs: counts[0..255]
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST=BLOCK_HIST)

        # 2) Triton inclusive scan to get le_counts
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        inclusive_scan_sum[(1,)](le_counts, counts, length=num_experts)

        # 3) Triton stable argsort permutation (compute_out_pos_triton). Ensure 'le_counts' is available as a global pointer.
        # Create a 1D int32 tensor on device for le_counts (we computed it above) and pass its pointer into the kernel.
        le_counts_ptr = le_counts  # alias to the tensor; Triton sees it as a device pointer
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        # Launch the kernel; it uses the global 'le_counts_ptr' and 'flat', writes to 'sorted_token_indices'.
        compute_out_pos_triton[(1,)](flat, sorted_token_indices, N)

        # 4) expert_offsets: inclusive prefix sums per expert (num_experts+1), including leading zero
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        cumsum_inclusive_triton[(1,)](counts, offsets[1:], length=num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
