import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(x_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Triton kernel to compute histogram of values in x_ptr[0:N].
    Values are expected to be in [0, num_buckets-1] (here num_buckets=256).
    Each program processes one element and performs an atomic add to the corresponding bucket.
    """
    i = tl.program_id(0)
    # We launch with grid (N,), so i ranges from 0 to N-1
    if i < N:
        v = tl.load(x_ptr + i)
        # Ensure v is in range and cast to int32 index
        # (We assume x_ptr contains int32 or int64; cast to int32 for indexing histogram.)
        v = v.to(tl.int32)
        # Guard against out-of-range values by clamping to [0, num_buckets-1]
        v = tl.maximum(v, 0)
        v = tl.minimum(v, num_buckets - 1)
        tl.atomic_add(histogram_ptr + v, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sum of histogram into out_ptr[1:].
    out_ptr[0] is not used. We do a simple sequential loop per program. For num_buckets=256,
    using a single program is fine. For larger num_buckets, we could implement a parallel scan,
    but here 256 is fixed by the original code.
    """
    # Initialize out[1:] = histogram[:]
    for b in range(num_buckets):
        # out_ptr is int32; histogram_ptr is int32
        # We read histogram[b] and write to out[b+1]
        # Using scalar loads/stores to set each position.
        # Triton's scalar loops are fine here.
        hist = tl.load(histogram_ptr + b)
        tl.store(out_ptr + b + 1, hist)
    # Inclusive scan: out[b+1] += out[b]
    for b in range(1, num_buckets):  # start from 1 and add previous
        prev = tl.load(out_ptr + b)
        cur = tl.load(out_ptr + b + 1)
        cur += prev
        tl.store(out_ptr + b + 1, cur)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Compute sorted_token_indices using torch.sort(stable=True) to ensure exact behavior.
        - Compute expert_offsets using Triton histogram + prefix sum.
        Returns:
          sorted_token_indices: 1D int64 tensor of length N
          expert_offsets: 1D int64 tensor of length (num_experts + 1), num_experts=256
        """
        # 1) Flatten and stable sort using PyTorch for correctness
        flat = topk_idx.reshape(-1)  # dtype is int32 as provided by get_inputs
        # Ensure dtype is int64 for sorted_token_indices to match torch.sort default index dtype
        # torch.sort returns indices as int64 by default; we keep that.
        sorted_token_indices = torch.sort(flat, stable=True).indices  # shape: [N], dtype: int64

        # 2) Compute histogram of expert IDs in Triton
        N = flat.numel()
        device = flat.device
        num_experts = 256  # matches the original code's hardcoded num_experts

        # Cast flat to int32 for histogram kernel (kernel assumes int32 indices [0..255])
        flat_i32 = flat.to(torch.int32)

        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        grid_hist = (N,)  # one program per element; atomic_add per element
        _histogram_kernel[grid_hist](flat_i32, N, histogram, num_buckets=num_experts)

        # 3) Compute prefix sum (cumulative counts) using Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # We set offsets[0] to 0 explicitly and compute inclusive scan in kernel
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        # Return offsets as int64 to match original behavior
        offsets = offsets.to(torch.int64)

        return sorted_token_indices, offsets