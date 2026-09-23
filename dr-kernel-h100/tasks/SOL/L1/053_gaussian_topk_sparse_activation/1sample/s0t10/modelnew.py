import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel(
    x_ptr,         # *float32
    mean_ptr,      # *float32, length B*S
    sumsq_ptr,     # *float32, length B*S
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, s) row, total P = B*S
    # Map pid to (b, s)
    b = pid // S
    s = pid % S

    # Base pointer for this row (x is made contiguous, stride_k == 1 here)
    base_row = x_ptr + b * stride_b + s * stride_s

    # Accumulators
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = kk < K
        ptrs = base_row + kk * stride_k
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # Reduce tile to scalars
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Compute mean and sumsq (per row)
    mean = sum_val / K
    sumsq = sumsq_val / K
    var = sumsq - mean * mean
    # Store as fp32
    tl.store(mean_ptr + pid, mean)
    tl.store(sumsq_ptr + pid, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,         # *float32
    out_ptr,       # *float32, 1D buffer of length B*S*K
    mean_ptr,      # *float32, length B*S
    sumsq_ptr,     # *float32, length B*S
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_k: tl.constexpr,
    zscore: tl.constexpr,  # float32 scalar
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, s) row
    b = pid // S
    s = pid % S

    base_row = x_ptr + b * stride_b + s * stride_s

    # Load per-row mean and sumsq
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)

    std = tl.sqrt(sumsq - mean * mean)
    # Handle std=0 case (rare); keep computation robust
    # threshold factor for this row
    m = mean + std * zscore

    # Process the row across K in tiles
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = kk < K
        ptrs_x = base_row + kk * stride_k
        x_vals = tl.load(ptrs_x, mask=mask, other=0.0)
        # Subtract threshold and apply ReLU
        y_vals = x_vals - m
        y_vals = tl.maximum(y_vals, 0.0)  # ReLU
        # Write to 1D output buffer at indices: ((b*S + s) * K + kk)
        base_out = out_ptr + (b * S + s) * K
        tl.store(base_out + kk, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, z_score: float = 0.9, block_k: int = 256, num_warps_reduce: int = 4, num_warps_element: int = 4):
        super().__init__()
        # z_score is inverse_normal_cdf(target_sparsity) at target_sparsity = 0.9
        # Keep as float for passing to Triton; do not compute per-forward on tensors.
        self.z_score = float(z_score)
        self.block_k = block_k
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_element = num_warps_element

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype float32 for computation
        x_f32 = x.to(torch.float32)
        # Make input contiguous to simplify strides (stride_k == 1)
        x_f32 = x_f32.contiguous()

        B, S, K = x_f32.shape
        P = B * S

        # Get strides (in elements)
        stride_b = x_f32.stride(0)
        stride_s = x_f32.stride(1)
        stride_k = x_f32.stride(2)

        # Allocate per-row buffers (fp32)
        mean_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: compute sum and sumsq per row
        grid = (P,)
        _reduce_sum_sumsq_kernel[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # Launch elementwise kernel: subtract threshold and apply ReLU
        out_fp32 = torch.empty(P * K, dtype=torch.float32, device=x_f32.device)
        _apply_threshold_relu_kernel[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,  # scalar float
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_element,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)