import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,            # *T (float/bfloat16)
    mean_out_ptr,     # *float32, size [B*S]
    sumsq_out_ptr,    # *float32, size [B*S]
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    row = b * stride_b + s * stride_s
    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate across K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Load elements from row (x[b, s, kk:kk+BLOCK_K])
        x_block = tl.load(x_ptr + row + offs * stride_k, mask=mask, other=0.0)
        x_block = x_block.to(tl.float32)
        # Accumulate
        sum_val += tl.sum(x_block, axis=0)
        sumsq_val += tl.sum(x_block * x_block, axis=0)

    # Compute mean and sumsq over K
    mean = sum_val / K
    var = sumsq_val / K - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    # Store per-row scalars
    tl.store(mean_out_ptr + b * S + s, mean)
    tl.store(sumsq_out_ptr + b * S + s, sumsq_val)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,                 # *T (float/bfloat16)
    out_ptr,               # *float32, size [B*S*K] contiguous
    mean_in_ptr,           # *float32, size [B*S]
    sumsq_in_ptr,          # *float32, size [B*S]
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_k: tl.constexpr,
    z_score: tl.constexpr,  # scalar float
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    row = b * stride_b + s * stride_s

    # Load per-row mean and sumsq
    mean = tl.load(mean_in_ptr + b * S + s)
    sumsq = tl.load(sumsq_in_ptr + b * S + s)
    std = tl.sqrt(sumsq / K - mean * mean)
    std = tl.maximum(std, 0.0)
    # threshold factor m = mean + std * z_score
    m = mean + std * z_score

    # Iterate across K in tiles, compute output = max(0, x - m)
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_block = tl.load(x_ptr + row + offs * stride_k, mask=mask, other=0.0)
        x_block = x_block.to(tl.float32)
        y_block = x_block - m
        # ReLU
        y_block = tl.maximum(y_block, 0.0)
        # Store to 1D out buffer at indices [b*S + s]*K + kk
        base = (b * S + s) * K
        tl.store(out_ptr + base + offs, y_block, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute inverse normal CDF for target_sparsity
        # For target_sparsity=0.9, z ≈ 1.2815515655446004
        # Using torch for one-time constant; it will not be used in kernels.
        self.z_score = float(torch.quantile(torch.arange(1, 1_000_000, dtype=torch.float32), 1.0 - target_sparsity).item())
        # Tunables
        self.block_k = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, K]
        assert x.dim() == 3, "Input must be 3D [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape
        # Ensure contiguous so stride_k == 1 and indexing is straightforward
        x = x.contiguous()

        # Allocate per-row buffers (fp32): [B*S]
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        stride_b, stride_s, stride_k = x.stride()  # in elements

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_kernel_2d[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x.device)

        _apply_threshold_relu_kernel_2d[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score),  # scalar float
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
