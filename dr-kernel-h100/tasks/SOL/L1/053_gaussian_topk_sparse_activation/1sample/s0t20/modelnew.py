import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,            # *const float, input tensor (contiguous)
    mean_out_ptr,     # *float, output per-row mean
    sumsq_out_ptr,    # *float, output per-row sumsq
    B, S, K,          # int32 dimensions
    stride_b, stride_s, stride_k,  # int32 strides
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)
    row = b * S + s

    # Guard in case grid > B*S
    if row >= B * S:
        return

    # Accumulate sum and sumsq over K in fp32
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq_ = tl.zeros((), dtype=tl.float32)

    # Base pointer for this row: offset = b*stride_b + s*stride_s
    base = b * stride_b + s * stride_s

    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)

    mean = sum_ / K
    # Population std: var = E[x^2] - (E[x])^2, std = sqrt(max(var, 0))
    var = sumsq_ / K - mean * mean
    std = tl.sqrt(tl.maximum(var, 0.0))

    # Store per-row mean and sumsq
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, sumsq_)


@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,            # *const float, input tensor (contiguous)
    out_ptr,          # *float, 1D output buffer of size B*S*K
    mean_in_ptr,      # *const float, per-row mean
    sumsq_in_ptr,     # *const float, per-row sumsq
    B, S, K,          # int32 dimensions
    stride_b, stride_s, stride_k,  # int32 strides
    z_score,          # float32 scalar (inverse normal cdf of target_sparsity)
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)
    row = b * S + s

    if row >= B * S:
        return

    # Load per-row statistics
    mean = tl.load(mean_in_ptr + row)
    sumsq = tl.load(sumsq_in_ptr + row)
    std = tl.sqrt(tl.maximum(sumsq / K - mean * mean, 0.0))
    threshold = mean + std * z_score

    base = b * stride_b + s * stride_s
    out_base = row * K

    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + out_base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Precompute z_score for target_sparsity (inverse normal cdf)
        # Use standard approximation; no Triton needed here
        from scipy.stats import norm
        self.z_score = float(norm.ppf(self.target_sparsity))
        self.block_k = int(block_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure we operate on CUDA tensors
        if not x.is_cuda:
            # Fallback to PyTorch if not on GPU (though evaluation uses CUDA)
            x_f32 = x.to(torch.float32).contiguous()
            B, S, K = x_f32.shape
            mean = x_f32.mean(dim=-1, keepdim=True)
            # population std: divide by K
            std = x_f32.pow(2).mean(dim=-1, keepdim=True) ** 0.5 - mean.pow(2) ** 0.5  # dummy to keep shape, use correct below
            # Correct std: var = E[x^2] - (E[x])^2
            mean = x_f32.mean(dim=-1, keepdim=True)
            sumsq = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
            var = sumsq - mean * mean
            std = (var.clamp_min(0.0)) ** 0.5
            threshold = mean + std * self.z_score
            sparse = torch.relu(x_f32 - threshold)
            return sparse.to(torch.bfloat16)

        # Triton path: require CUDA
        x_f32 = x.to(torch.float32).contiguous()
        B, S, K = x_f32.shape
        stride_b, stride_s, stride_k = x_f32.stride()

        # Per-row buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

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
            float(self.z_score),  # scalar float
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)