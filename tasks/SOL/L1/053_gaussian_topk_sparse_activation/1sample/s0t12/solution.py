import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel(
    x_ptr,               # *fp32
    mean_ptr,            # *fp32, size B*S
    sumsq_ptr,           # *fp32, size B*S
    B: tl.constexpr,     # int
    S: tl.constexpr,     # int
    K: tl.constexpr,     # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_k: tl.constexpr,  # int
    tiles_k: tl.constexpr,   # int = ceil_div(K, BLOCK_K)
    BLOCK_K: tl.constexpr,   # int
):
    # Grid is (B, S, tiles_k)
    b = tl.program_id(0)
    s = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Compute base pointer for the start of this (b, s) row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate over this tile only
    kk = tile_id * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = kk < K

    # Load the tile
    x_ptrs = x_ptr + base + kk * stride_k
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

    # Accumulate sum and sum of squares
    total_sum += tl.sum(x_vals, axis=0)
    total_sumsq += tl.sum(x_vals * x_vals, axis=0)

    # Store per-row results
    # index in mean/sumsq is b*S + s
    idx = b * S + s
    tl.store(mean_ptr + idx, total_sum)
    tl.store(sumsq_ptr + idx, total_sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,               # *fp32
    out_ptr,             # *fp32, 1D buffer of size B*S*K
    mean_ptr,            # *fp32, size B*S
    sumsq_ptr,           # *fp32, size B*S
    B: tl.constexpr,     # int
    S: tl.constexpr,     # int
    K: tl.constexpr,     # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_k: tl.constexpr,  # int
    tiles_k: tl.constexpr,   # int
    BLOCK_K: tl.constexpr,   # int
    z_score: tl.constexpr,   # float
):
    # Grid is (B, S, tiles_k)
    b = tl.program_id(0)
    s = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Base pointer for this (b, s) row
    base = b * stride_b + s * stride_s

    # Load per-row stats
    idx = b * S + s
    mean = tl.load(mean_ptr + idx)
    sumsq = tl.load(sumsq_ptr + idx)
    # Compute variance and std
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to fp rounding
    std = tl.sqrt(var)

    # Compute per-row threshold factor m = mean + std * z_score
    m = mean + std * z_score

    # Iterate over this tile only
    kk = tile_id * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = kk < K

    # Load x tile
    x_ptrs = x_ptr + base + kk * stride_k
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

    # Apply y = max(x - m, 0)
    y = x_vals - m
    y = tl.maximum(y, 0.0)  # ReLU

    # Store to output (contiguous 1D)
    # Linear index for this row is ((b*S + s) * K) + kk
    row_base_out = (b * S + s) * K
    out_ptrs = out_ptr + row_base_out + kk
    tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity) once
        # Use torch for initialization; forward uses only Triton kernels.
        import torch
        self.z_score = float(torch.distributions.normal.InvProb.default_cumulative_distribution.invprob(torch.tensor(target_sparsity)).item())
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input of shape [B, S, K]
        assert x.ndim == 3, f"Expected 3D input [B, S, K], got shape {x.shape}"
        B, S, K = x.shape

        # Make input contiguous (float32) to simplify stride_k handling
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()  # after .contiguous(), stride_k == 1

        # Number of tiles along K
        tiles_k = triton.cdiv(K, self.block_k)

        # Allocate per-row stats (fp32) of size B*S
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: grid over (B, S, tiles_k)
        grid = (B, S, tiles_k)
        _reduce_sum_sumsq_kernel[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            tiles_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Launch elementwise kernel: same grid
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_f32.device)
        _apply_threshold_relu_kernel[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            tiles_k,
            self.block_k,
            self.z_score,
            num_warps=4,
            num_stages=2,
        )

        # Reshape to [B, S, K] and return in bfloat16 to match original behavior
        out_fp32 = out_fp32.view(B, S, K)
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
