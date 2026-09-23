import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(flat_ptr + offsets, mask=mask, other=0)
    x = x.to(tl.int32)
    valid = (x >= 0) & (x < num_experts) & mask
    tl.atomic_add(counts_ptr + x, 1, mask=valid)


@triton.jit
def inclusive_scan_inplace(counts_ptr, offsets_ptr, num_experts: tl.int32, LOG: tl.constexpr):
    # Hillis–Steele inclusive scan implemented as a fixed number of passes over the 1D array.
    # counts_ptr[0..num_experts-1] must be valid int32.
    for i in range(LOG):
        stride = 1 << i
        carry = tl.zeros((), dtype=tl.int32)
        for j in range(0, num_experts):
            current = tl.load(counts_ptr + j)
            # For masked loads beyond num_experts, we don't have values; but counts_ptr is num_experts-sized.
            # Instead, we only update valid positions (0..num_experts-1) in offsets_ptr.
            # Add carry to current only when j >= stride; for j < stride, the previous iteration added the carry.
            tl.store(offsets_ptr + j, current + carry)
            if (j + stride) < num_experts:
                carry = carry + tl.load(counts_ptr + (j + stride))
            else:
                carry = carry + 0
        # After the loop, carry contains the sum of counts[0..num_experts-1].


@triton.jit
def stable_bitonic_sort_inplace(vals_ptr, idx_ptr, N: tl.int32, BLOCK: tl.int32, LOG: tl.int32):
    # Bitonic sort network over BLOCK lanes, sorting by (value, index) lexicographically.
    # vals_ptr and idx_ptr are arrays of length BLOCK; we only use first N lanes.
    # Padded lanes (N..BLOCK-1) are set to sentinel MAX_INT, ensuring they sort to the end.
    for k in range(1, LOG + 1):
        size = 1 << k
        for j in range(k - 1, -1, -1):
            i = 1 << j
            partner = i - 1  # local index relative to k
            ixj = partner - (partner & i)  # distance within k
            ix = size // 2
            while ix >= 1:
                ix = ix // 2
                other = (partner ^ ix)
                a = partner & ix
                ascend = ( (partner & size) == 0 )
                x_id = partner
                y_id = other

                # Load current values and indices
                val_x = tl.load(vals_ptr + x_id)
                idx_x = tl.load(idx_ptr + x_id)
                val_y = tl.load(vals_ptr + y_id)
                idx_y = tl.load(idx_ptr + y_id)

                # Stable comparison: (val_x, idx_x) vs (val_y, idx_y)
                cmp_v = val_x > val_y
                cmp_i = idx_x > idx_y
                gt = (val_x > val_y) | ((val_x == val_y) & (idx_x > idx_y))

                # Swap if needed
                if (ascend and gt) or (not ascend and not gt):
                    tmp_val = val_x
                    tmp_idx = idx_x
                    val_x = val_y
                    idx_x = idx_y
                    val_y = tmp_val
                    idx_y = tmp_idx

                # Store back
                tl.store(vals_ptr + x_id, val_x)
                tl.store(idx_ptr + x_id, idx_x)
                tl.store(vals_ptr + y_id, val_y)
                tl.store(idx_ptr + y_id, idx_y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure contiguous and device
        device = topk_idx.device
        num_experts = 256  # matches original run behavior

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        dtype = flat.dtype  # int32 by construction

        # Prepare outputs
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Triton histogram kernel
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024  # tuneable block size
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](flat, counts, N, num_experts, BLOCK)

        # Triton inclusive scan to get expert_offsets (prefix sums)
        LOG = 8  # since num_experts = 256 -> 2^8
        # Copy counts into offsets buffer and run scan in-place
        offsets_scan = torch.empty(num_experts, dtype=torch.int32, device=device)
        offsets_scan.copy_(counts)  # initialize with counts
        # Launch inclusive scan kernel (fixed number of passes)
        # We need grid size 1 for 1D array; Triton can handle scalar launch grid=(1,)
        inclusive_scan_inplace[(1,)](offsets_scan, offsets_scan, num_experts, LOG)

        # Fill expert_offsets: offsets[i] = sum of counts[0..i], offsets[num_experts] = N
        expert_offsets[:num_experts] = offsets_scan
        expert_offsets[-1] = N

        # Triton bitonic sort for argsort (sorted_token_indices)
        # Choose BLOCK_SORT as next power of two >= N, cap at 4096
        BLOCK_SORT = 1
        while BLOCK_SORT < N:
            BLOCK_SORT *= 2
        BLOCK_SORT = min(BLOCK_SORT, 4096)
        LOG_SORT = 0
        while (1 << LOG_SORT) < BLOCK_SORT:
            LOG_SORT += 1

        # Prepare values and indices buffers
        MAX_INT = (1 << 31) - 1
        vals = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)

        # Initialize vals and idx_out
        vals[:N] = flat
        vals[N:] = MAX_INT
        idx_out[:N] = torch.arange(N, device=device)
        idx_out[N:] = torch.zeros(BLOCK_SORT - N, dtype=torch.int32, device=device)

        # Launch stable bitonic sort
        stable_bitonic_sort_inplace[(1,)](vals, idx_out, N, BLOCK_SORT, LOG_SORT)

        # sorted_token_indices is the first N entries of idx_out
        sorted_token_indices = idx_out[:N]

        return sorted_token_indices, expert_offsets