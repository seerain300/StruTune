import math
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,           # *fp32, input [B, S, K] contiguous
    mean_out_ptr,    # *fp32, output [P] where P=B*S
    sumsq_out_ptr,   # *fp32, output [P] where P=B*S
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b, stride_s, stride_k,      # int strides for x (assumed contiguous: stride_k == 1)
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Tile over K
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        # x_ptr is fp32 and contiguous along K
        x_vals = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        acc_sum += tl.sum(x_vals, axis=0)
        acc_sumsq += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and sumsq (per element counts are 1 per output, but we need per-K)
    mean = acc_sum / K
    sumsq = acc_sumsq / K

    # Store per-row mean and sumsq
    pid = b * S + s
    tl.store(mean_out_ptr + pid, mean)
    tl.store(sumsq_out_ptr + pid, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,            # *fp32, input [B, S, K] contiguous
    out_ptr,          # *fp32, output [P*K] contiguous
    mean_in_ptr,      # *fp32, input [P] means
    sumsq_in_ptr,     # *fp32, input [P] sum of squares
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b, stride_s, stride_k,      # int strides for x (assumed contiguous: stride_k == 1)
    z_score,                                   # scalar float (inverse normal cdf at target_sparsity)
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Load per-row mean and sumsq
    pid = b * S + s
    mean = tl.load(mean_in_ptr + pid)
    sumsq = tl.load(sumsq_in_ptr + pid)

    # Compute std (population std): var = E[x^2] - (E[x])^2
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # Threshold factor per row
    threshold = mean + std * z_score

    # Apply elementwise: out = max(0, x - threshold)
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_vals = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        y = x_vals - threshold
        y = tl.where(y > 0.0, y, 0.0)  # ReLU
        # Store into contiguous 1D output: index offset for this row is pid*K + kk
        out_idx = pid * K + kk
        tl.store(out_ptr + out_idx + tl.arange(0, BLOCK_K), y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute z_score for target_sparsity (inverse normal cdf). Use PyTorch for accuracy.
        # torch.distributions.Normal(0, 1).icdf(0.9) ≈ 1.2815515655446004
        # Here we compute once.
        import scipy.stats
        self.z_score = float(scipy.stats.norm.ppf(target_sparsity))  # scalar float
        # Triton tuning parameters
        self.block_k = 256
        self.num_warps_reduce = 4
        self.num_warps_elem = 4

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure we operate in fp32 for numerics; original returns bfloat16.
        # Move to device, make contiguous for Triton.
        x_f32 = x.to(torch.float32).contiguous()
        B, S, K = x_f32.shape
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
            num_warps=self.num_warps_reduce,
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
            self.z_score,  # scalar float
            self.block_k,
            num_warps=self.num_warps_elem,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
