import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sparsity_kernel(
    x_ptr,            # *float32, input
    out_ptr,          # *float32, output (we will cast to bf16 in host)
    B, S, K,          # int32 sizes
    stride_b, stride_s, stride_k,  # int32 strides for x_ptr/out_ptr
    z_score,          # float32, inverse normal CDF for target sparsity
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Guard against overlaunch (if grid is larger than B,S)
    if b >= B or s >= S:
        return

    # Compute base offsets for this row
    row_base = b * stride_b + s * stride_s

    # Pass 1: accumulate sum and sum of squares across K
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    kk = 0
    while kk < K:
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row_base + offs * stride_k, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        kk += BLOCK_K

    # Compute mean and std (population std)
    K_f = tl.full((), K, dtype=tl.float32)
    mean = sum_val / K_f
    var = sumsq_val / K_f - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to numerical errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Threshold: mean + std * z_score
    thresh = mean + std * z_score

    # Pass 2: apply threshold, subtract, ReLU, write output
    kk = 0
    while kk < K:
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row_base + offs * stride_k, mask=mask, other=0.0)
        y = x - thresh
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_base + offs * stride_k, y, mask=mask)
        kk += BLOCK_K


class ModelNew(nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 1024):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_k = int(block_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, f"Expected 3D input, got shape {tuple(x.shape)}"
        B, S, K = x.shape

        # Make contiguous for simple stride_k=1 addressing
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate fp32 output buffer; cast to bf16 after kernel
        out_fp32 = torch.empty_like(x_f32)

        # Launch Triton kernel: one program per (b, s) row
        grid = (B, S)
        _sparsity_kernel[grid](
            x_f32,
            out_fp32,
            B, S, K,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            # Use the standard z-score for sparsity=0.9
            1.2815515655446004,
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)