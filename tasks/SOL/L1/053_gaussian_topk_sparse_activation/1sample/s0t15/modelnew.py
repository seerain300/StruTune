import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel(
    x_ptr,
    mean_ptr,
    sumsq_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row (b, s)
    pid = tl.program_id(0)
    # Compute b and s from pid (grid is 1D over B*S, but we treat as 2D here implicitly)
    # Note: in the launch, we use a 2D grid (B, S) so pid corresponds directly to row index.
    # To be robust, we can derive b and s by integer division/modulo on a combined index.
    # However, Triton does not directly support passing 2D grid ids here. We instead pass B,S
    # and rely on the host to map pid to (b, s). We use a 2D grid launch in forward.
    pass  # placeholder, actual implementation below


# Below are the corrected kernels using a 2D grid (B, S) so b and s are directly available


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,
    mean_ptr,
    sumsq_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    # Base pointer for this row
    base = b * stride_b + s * stride_s
    # Accumulators in fp32
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Tile across K
    for kk in range(0, K, BLOCK_K):
        offsets = kk + tl.arange(0, BLOCK_K)
        mask = offsets < K
        vals = tl.load(x_ptr + base + offsets * stride_k, mask=mask, other=0.0)
        acc_sum += tl.sum(vals, axis=0)
        acc_sumsq += tl.sum(vals * vals, axis=0)

    mean = acc_sum / K
    sumsq = acc_sumsq / K  # population sum of squares
    # Store per-row mean and sumsq
    row_id = b * S + s
    tl.store(mean_ptr + row_id, mean)
    tl.store(sumsq_ptr + row_id, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,
    out_ptr,
    mean_ptr,
    sumsq_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    z_score,  # scalar float
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    base = b * stride_b + s * stride_s
    # Load mean and sumsq for this row
    row_id = b * S + s
    mean = tl.load(mean_ptr + row_id)
    sumsq = tl.load(sumsq_ptr + row_id)
    # Compute std
    var = sumsq - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to round-off
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Threshold factor m = mean + std * z_score
    m = mean + std * z_score
    # Apply: out = max(x - m, 0)
    for kk in range(0, K, BLOCK_K):
        offsets = kk + tl.arange(0, BLOCK_K)
        mask = offsets < K
        x_vals = tl.load(x_ptr + base + offsets * stride_k, mask=mask, other=0.0)
        out_vals = x_vals - m
        out_vals = tl.maximum(out_vals, 0.0)  # ReLU
        tl.store(out_ptr + row_id * K + offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score once using torch (host-only, not per-forward tensor op)
        self.z_score = torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(target_sparsity))
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, K]
        assert x.dim() == 3, "Input must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape

        # Make input contiguous and in fp32 for stable stats
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate per-row buffers (fp32)
        P = B * S
        mean_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row using 2D grid
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
            float(self.z_score.item()),
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)