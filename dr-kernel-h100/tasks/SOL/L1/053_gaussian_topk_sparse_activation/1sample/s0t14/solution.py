import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_mean_sumsq_kernel(
    x_ptr,          # *fp32, input tensor pointer (contiguous)
    mean_ptr,       # *fp32, per-row mean (size B*S)
    sumsq_ptr,      # *fp32, per-row sumsq (size B*S)
    B: tl.int32, S: tl.int32, K: tl.int32,
    stride_b: tl.int32, stride_s: tl.int32, stride_k: tl.int32,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # 0..B*S-1
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    total = 0.0
    sumsq = 0.0

    # Loop over K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        ptrs = x_ptr + base + offs * stride_k
        x = tl.load(ptrs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        sumsq += tl.sum(x * x, axis=0)

    mean = total / K
    sqavg = sumsq / K
    # Write per-row stats
    mean_ptr[pid] = mean
    sumsq_ptr[pid] = sqavg


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,          # *fp32, input tensor pointer (contiguous)
    out_ptr,        # *fp32, 1D output buffer (size B*S*K)
    mean_ptr,       # *fp32, per-row mean (size B*S)
    sumsq_ptr,      # *fp32, per-row sumsq (size B*S)
    B: tl.int32, S: tl.int32, K: tl.int32,
    stride_b: tl.int32, stride_s: tl.int32, stride_k: tl.int32,
    tiles_k: tl.int32,
    zscore: tl.float32,  # scalar
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, S, tiles_k)
    b = tl.program_id(0)
    s = tl.program_id(1)
    tile = tl.program_id(2)

    pid_row = b * S + s
    # Load per-row mean and sumsq
    mean = mean_ptr[pid_row]
    sumsq = sumsq_ptr[pid_row]
    std = tl.sqrt(sumsq - mean * mean)

    # Compute threshold factor for this row
    m = mean + std * zscore

    # Compute base offsets
    base_in = b * stride_b + s * stride_s
    base_out = pid_row * (K * tiles_k)

    # Each program handles one tile along K
    kk_start = tile * BLOCK_K
    offs = kk_start + tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load x, subtract m, apply ReLU, store
    in_ptrs = x_ptr + base_in + offs * stride_k
    x = tl.load(in_ptrs, mask=mask, other=0.0)
    y = x - m  # subtract per-row threshold
    # ReLU: max(y, 0)
    y = tl.maximum(y, 0.0)
    out_ptrs = out_ptr + base_out + tile * BLOCK_K + offs
    tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256, num_warps_reduce: int = 4, num_warps_element: int = 4):
        super().__init__()
        # Compute z_score once using torch (allowed for init); forward is Triton-only
        self.z_score = torch.tensor(target_sparsity, dtype=torch.float32).apply(lambda p: torch.distributions.normal.Normal(0, 1).icdf(1 - p))
        # Triton tiling parameters
        self.block_k = block_k
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_element = num_warps_element

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is contiguous and in float32 for computation
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, K = x_f32.shape
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate per-row stats (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid_reduce = (B * S,)
        _reduce_mean_sumsq_kernel[grid_reduce](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # Prepare output buffer (1D: B*S*K)
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_f32.device)

        # Compute tiles along K
        tiles_k = (K + self.block_k - 1) // self.block_k

        # Launch elementwise kernel: 3D grid over (B, S, tiles_k)
        grid_element = (B, S, tiles_k)
        _apply_threshold_relu_kernel[grid_element](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            tiles_k,
            float(self.z_score.item()),  # pass scalar zscore
            self.block_k,
            num_warps=self.num_warps_element,
            num_stages=2,
        )

        # Reshape to [B, S, K] and return in bfloat16
        out_fp32 = out_fp32.view(B, S, K)
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
