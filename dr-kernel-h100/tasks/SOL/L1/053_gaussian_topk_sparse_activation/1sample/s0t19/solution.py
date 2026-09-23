import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,              # *fp32
    mean_row_ptr,       # *fp32 (length B*S)
    sumsq_row_ptr,      # *fp32 (length B*S)
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b, stride_s, stride_k,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over (b, s), one program per row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Guard for out-of-range (if grid > B*S, though we set grid to (B, S))
    if b >= B or s >= S:
        return

    # Base pointer for this row
    row_ptr = x_ptr + b * stride_b + s * stride_s

    # Accumulate sum and sumsq in fp32
    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate across K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        vals = tl.load(row_ptr + offs * stride_k, mask=mask, other=0.0)
        # Reduce within the tile
        total_sum += tl.sum(vals, axis=0)
        total_sumsq += tl.sum(vals * vals, axis=0)

    # Compute mean and sumsq per row (scalar outputs)
    K_fp = tl.cast(K, tl.float32)
    mean = total_sum / K_fp
    sumsq = total_sumsq / K_fp

    # Write per-row scalars
    idx = b * S + s
    tl.store(mean_row_ptr + idx, mean)
    tl.store(sumsq_row_ptr + idx, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,              # *fp32
    out_ptr,            # *fp32 (length B*S*K)
    mean_row_ptr,       # *fp32 (length B*S)
    sumsq_row_ptr,      # *fp32 (length B*S)
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b, stride_s, stride_k,
    z_score,            # scalar float
    BLOCK_K: tl.constexpr,
):
    # 2D grid over (b, s), one program per row
    b = tl.program_id(0)
    s = tl.program_id(1)

    if b >= B or s >= S:
        return

    # Base pointer for this row
    row_ptr = x_ptr + b * stride_b + s * stride_s
    out_row_ptr = out_ptr + (b * S + s) * K

    # Load mean and sumsq for this row
    idx = b * S + s
    mean = tl.load(mean_row_ptr + idx)
    sumsq = tl.load(sumsq_row_ptr + idx)

    # Compute std
    K_fp = tl.cast(K, tl.float32)
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)  # avoid small negative due to roundoff
    std = tl.sqrt(var)

    # Per-row threshold factor m = mean + std * z
    m = mean + std * z_score

    # Apply: out = max(x - m, 0) across K
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_vals = tl.load(row_ptr + offs * stride_k, mask=mask, other=0.0)
        y = x_vals - m
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_row_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity) once
        # Use torch's normal inverse CDF for scalar init
        self.z_score = torch.normal.icdf(torch.tensor(target_sparsity, dtype=torch.float32))  # deprecated: use torch.distributions.normal
        # Fallback: manual mapping for 0.9
        if target_sparsity == 0.9:
            self.z_score = torch.tensor(1.2815515655446004, dtype=torch.float32)
        else:
            # For other sparsities, use torch if available; otherwise, we rely on Triton side to pass scalar
            self.z_score = torch.tensor(torch.distributions.normal.Normal(0, 1).icdf(target_sparsity).item() if hasattr(torch, 'distributions') and hasattr(torch.distributions.normal, 'Normal') else 0.0, dtype=torch.float32)
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, K]
        B, S, K = x.shape

        # Ensure fp32 and contiguous for predictable strides
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate per-row buffers (fp32)
        P = B * S
        mean_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_kernel_2d[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_f32.device)

        _apply_threshold_relu_kernel_2d[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score.item()),  # scalar float
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
