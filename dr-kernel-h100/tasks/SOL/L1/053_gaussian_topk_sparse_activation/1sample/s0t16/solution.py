import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,                 # *fp32
    mean_out_ptr,          # *fp32, shape [B*S]
    sumsq_out_ptr,         # *fp32, shape [B*S]
    B, S, K,               # int32
    stride_b, stride_s, stride_k,  # int32 strides in elements
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D grid: one program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulate sum and sum of squares across K in tiles
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Load tile from row
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        # Accumulate (unbiased=False): sum and sumsq across the tile
        # Reduce vector to scalar
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute mean and sum of squares for the row
    mean = sum_val / K
    sumsq = sumsq_val / K

    # Write per-row results
    row_idx = b * S + s
    tl.store(mean_out_ptr + row_idx, mean)
    tl.store(sumsq_out_ptr + row_idx, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,                 # *fp32
    out_ptr,               # *fp32, shape [B*S*K] contiguous
    mean_ptr,              # *fp32, shape [B*S]
    sumsq_ptr,             # *fp32, shape [B*S]
    B, S, K,               # int32
    stride_b, stride_s, stride_k,  # int32 strides in elements
    z_score,               # fp32 scalar (inverse normal CDF of target sparsity)
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D grid: one program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Load per-row statistics
    row_idx = b * S + s
    mean = tl.load(mean_ptr + row_idx)
    sumsq = tl.load(sumsq_ptr + row_idx)
    # std = sqrt(sumsq - mean^2)
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Compute threshold factor: m = mean + std * z_score
    m = mean + std * z_score

    # Iterate across K in tiles, subtract m, apply ReLU, and store
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        y = x - m
        # ReLU
        y = tl.maximum(y, 0.0)
        # Store contiguous 1D output at [row_idx, kk] -> linear index
        out_base = row_idx * K
        tl.store(out_ptr + out_base + kk + tl.arange(0, BLOCK_K), y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256, num_warps_reduce: int = 4, num_warps_elem: int = 4):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Precompute inverse normal CDF once; not used in Triton per-call, but needed for threshold factor
        # Use torch to get a scalar float
        self.z_score = torch.normal.icdf(torch.tensor(self.target_sparsity))
        # Kernel launch parameters
        self.block_k = int(block_k)
        self.num_warps_reduce = int(num_warps_reduce)
        self.num_warps_elem = int(num_warps_elem)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, K] (dtype may vary; we convert to fp32 for compute)
        B, S, K = x.shape

        # Make input contiguous for simple, predictable strides
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate per-row buffers (fp32) on device
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
            num_warps=self.num_warps_reduce,
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
            num_warps=self.num_warps_elem,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
