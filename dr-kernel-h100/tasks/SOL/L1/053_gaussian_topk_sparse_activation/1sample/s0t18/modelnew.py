import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,            # *fp32, input pointer (contiguous: [B, S, K])
    mean_out_ptr,     # *fp32, per-row mean output buffer of length B*S
    sumsq_out_ptr,    # *fp32, per-row sumsq output buffer of length B*S
    B, S, K,          # int32 sizes
    stride_b, stride_s, stride_k,  # int32 strides (in elements)
    BLOCK_K: tl.constexpr,         # tile size across K
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Safety: if outside bounds, return (shouldn't happen with grid=(B,S))
    if b >= B or s >= S:
        return

    # Compute per-row base pointer (row is [b, s, :])
    row_base = b * stride_b + s * stride_s

    # Accumulate sum and sum of squares across K
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in tiles
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Load a tile of the row; assume x_ptr points to contiguous data
        vals = tl.load(x_ptr + row_base + offs * stride_k, mask=mask, other=0.0)
        # Reduce within the tile to scalars
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Compute mean and sumsq for this row
    # Note: mean = sum / K, sumsq is sum of squares / K (matches torch.mean/sum behavior when computing std)
    mean = sum_val / K
    # sumsq is the average of squares across K; var = sumsq - mean^2
    sumsq_avg = sumsq_val / K
    var = sumsq_avg - mean * mean
    # std = sqrt(max(var, 0)) to avoid tiny negative due to rounding
    std = tl.sqrt(tl.maximum(var, 0.0))

    # Write per-row results
    pid = b * S + s
    tl.store(mean_out_ptr + pid, mean)
    tl.store(sumsq_out_ptr + pid, sumsq_avg)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,            # *fp32, input pointer (contiguous: [B, S, K])
    out_ptr,          # *fp32, 1D output buffer of length (B*S*K)
    mean_ptr,         # *fp32, per-row mean buffer of length B*S
    sumsq_ptr,        # *fp32, per-row sumsq buffer of length B*S
    B, S, K,          # int32 sizes
    stride_b, stride_s, stride_k,  # int32 strides (in elements)
    z_score,          # fp32 scalar z = inverse_normal_cdf(target_sparsity)
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    row_base = b * stride_b + s * stride_s

    # Load per-row mean and sumsq
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq_avg = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(tl.maximum(sumsq_avg - mean * mean, 0.0))

    # Compute threshold factor: mean + std * z
    m = mean + std * z_score

    # Iterate across K in tiles and write ReLU(x - m)
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        vals = tl.load(x_ptr + row_base + offs * stride_k, mask=mask, other=0.0)
        res = vals - m
        # ReLU
        res = tl.maximum(res, 0.0)
        # Store into 1D output buffer at linear indices: pid*K + k + offs
        tl.store(out_ptr + pid * K + offs, res, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity) once
        # Using torch's built-in for correctness; not per-call tensor op.
        self.z_score = torch.tensor(torch.normal.icdf(torch.tensor(target_sparsity)), dtype=torch.float32)
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, K], arbitrary device
        B, S, K = x.shape

        # Ensure input is contiguous and in fp32 for numeric stability
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate per-row buffers (fp32) of length B*S
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
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(P * K, dtype=torch.float32, device=x_f32.device)

        _apply_threshold_relu_kernel_2d[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score.item()),  # pass scalar
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)