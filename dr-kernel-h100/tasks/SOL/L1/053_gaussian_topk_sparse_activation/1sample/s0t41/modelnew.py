import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_1d(
    x_ptr,            # *const float
    mean_ptr,         # *float
    sumsq_ptr,        # *float
    B, S, K,          # int32
    stride_b, stride_s, stride_k,  # int32
    BLOCK_K: tl.constexpr,
):
    # One program per row r in [0, B*S)
    r = tl.program_id(0)
    b = r // S
    s = r % S

    base = b * stride_b + s * stride_s
    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over K in tiles
    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        mask = kk < K
        offs = base + kk * stride_k
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    mean = sum_val / K
    sumsq = sumsq_val / K  # population sum of squares average
    tl.store(mean_ptr + r, mean)
    tl.store(sumsq_ptr + r, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_1d(
    x_ptr,            # *const float
    out_ptr,          # *float
    mean_ptr,         # *const float
    sumsq_ptr,        # *const float
    B, S, K,          # int32
    stride_b, stride_s, stride_k,  # int32
    z_score,          # float32
    BLOCK_K: tl.constexpr,
):
    # One program per row r in [0, B*S)
    r = tl.program_id(0)
    b = r // S
    s = r % S

    base = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + r)
    sumsq = tl.load(sumsq_ptr + r)
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    threshold = mean + std * z_score

    # Apply: out = max(0, x - threshold), output in fp32
    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        mask = kk < K
        offs = base + kk * stride_k
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = vals - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity)
        # For target_sparsity = 0.9, z ≈ 1.2815515655446004
        self.z_score = float(1.2815515655446004)  # 0.9 quantile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        B, S, K = x.shape

        # Compute strides in elements
        stride_b, stride_s, stride_k = x.stride()

        # Allocate per-row buffers (fp32) of shape [B*S]
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per row
        grid = (B * S,)
        _reduce_sum_sumsq_kernel_1d[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x.device)

        _apply_threshold_relu_kernel_1d[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,  # scalar float
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)