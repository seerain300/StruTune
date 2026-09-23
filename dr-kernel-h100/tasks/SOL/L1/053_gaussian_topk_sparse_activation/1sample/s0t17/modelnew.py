import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,              # *f32
    mean_ptr,           # *f32, length B*S
    sumsq_ptr,          # *f32, length B*S
    B: tl.int32,
    S: tl.int32,
    K: tl.int32,
    stride_b: tl.int32,
    stride_s: tl.int32,
    stride_k: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Accumulators (fp32 scalars)
    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate across K in tiles
    offs = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        kk = k_start + offs
        mask = kk < K
        # Compute base pointer for this row
        row_base = b * stride_b + s * stride_s
        # Load elements x[b, s, kk] as a vector; masked for kk >= K
        x_vals = tl.load(x_ptr + row_base + kk * stride_k, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        total_sum += tl.sum(x_vals, axis=0)
        total_sumsq += tl.sum(x_vals * x_vals, axis=0)

    # Store per-row mean and sumsq
    mean_ptr[b * S + s] = total_sum / K
    sumsq_ptr[b * S + s] = total_sumsq


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,              # *f32
    out_ptr,            # *f32, length (B*S*K)
    mean_ptr,           # *f32, length B*S
    sumsq_ptr,          # *f32, length B*S
    B: tl.int32,
    S: tl.int32,
    K: tl.int32,
    stride_b: tl.int32,
    stride_s: tl.int32,
    stride_k: tl.int32,
    z_score: tl.float32,  # scalar
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Compute mean and sumsq for this row
    row_index = b * S + s
    mean = mean_ptr[row_index]
    sumsq = sumsq_ptr[row_index]

    # Compute std; handle tiny negative due to rounding
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Threshold factor per row
    threshold = mean + std * z_score

    # Iterate across K in tiles; write to contiguous output buffer
    offs = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        kk = k_start + offs
        mask = kk < K
        row_base = b * stride_b + s * stride_s
        x_vals = tl.load(x_ptr + row_base + kk * stride_k, mask=mask, other=0.0)
        # Apply ReLU after subtracting threshold
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)
        # Store into 1D output: out[b*S*K + b*S*K + kk]
        out_idx = row_index * K + kk
        tl.store(out_ptr + out_idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score (inverse normal cdf) once; forward uses only Triton
        # Use torch to compute scalar; not per-call tensor op
        self.z_score = torch.normal.icdf(torch.tensor(target_sparsity, dtype=torch.float32))
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, K] tensor
        Returns: [B, S, K] tensor in bfloat16 with adaptive threshold ReLU applied.
        """
        # Ensure 3D input
        assert x.dim() == 3, "Input must be 3D [B, S, K]"
        B, S, K = x.shape

        # Make input contiguous and cast to fp32 for computation
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Per-row buffers (fp32)
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
            float(self.z_score.item()),  # pass as scalar float
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)