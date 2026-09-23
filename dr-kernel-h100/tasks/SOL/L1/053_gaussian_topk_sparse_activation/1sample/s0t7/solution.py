import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Reduction kernel: compute per-row sum and sum of squares across K features.
# One program per (b, s) row. It iterates over K in chunks of BLOCK_K, accumulates
# sum and sumsq in fp32, and writes mean_row[idx] and sumsq_row[idx] where idx = b*S + s.
if TRITON_AVAILABLE:
    @triton.jit
    def _reduce_sum_sumsq_kernel(
        x_ptr,                      # *fp32
        mean_ptr,                   # *fp32, size B*S
        sumsq_ptr,                  # *fp32, size B*S
        B: tl.constexpr,            # int
        S: tl.constexpr,            # int
        K: tl.constexpr,            # int
        stride_b: tl.constexpr,     # int
        stride_s: tl.constexpr,     # int
        stride_k: tl.constexpr,     # int
        BLOCK_K: tl.constexpr,      # int
    ):
        pid = tl.program_id(0)
        b = pid // S
        s = pid % S
        idx = b * S + s

        # Accumulators
        sum_val = 0.0
        sumsq_val = 0.0

        # Iterate over K in tiles
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            mask = kk < K
            # Compute pointer to x[b, s, kk]
            ptr = x_ptr + b * stride_b + s * stride_s + kk * stride_k
            vals = tl.load(ptr, mask=mask, other=0.0)
            vals = vals.to(tl.float32)
            sum_val += tl.sum(vals, axis=0)
            sumsq_val += tl.sum(vals * vals, axis=0)

        mean = sum_val / K
        var = sumsq_val - mean * mean
        # Clamp variance to non-negative to guard against tiny negative due to fp errors
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        # Store per-row stats
        tl.store(mean_ptr + idx, mean)
        tl.store(sumsq_ptr + idx, sumsq_val)  # we'll compute std in host from mean & sumsq
        # Note: we could also store std directly here if desired, but computing in host is fine.

# Elementwise kernel: subtract per-row threshold m = mean + std * z and apply ReLU.
# One program per (b, s) row; iterate over K in tiles, load mean/std for this row,
# compute m, subtract, apply ReLU, and store to out.
if TRITON_AVAILABLE:
    @triton.jit
    def _apply_threshold_relu_kernel(
        x_ptr,                      # *fp32
        out_ptr,                    # *fp32
        mean_ptr,                   # *fp32, size B*S
        sumsq_ptr,                  # *fp32, size B*S
        B: tl.constexpr,            # int
        S: tl.constexpr,            # int
        K: tl.constexpr,            # int
        stride_b: tl.constexpr,     # int
        stride_s: tl.constexpr,     # int
        stride_k: tl.constexpr,     # int
        zscore: tl.constexpr,       # float
        BLOCK_K: tl.constexpr,      # int
    ):
        pid = tl.program_id(0)
        b = pid // S
        s = pid % S

        # Load per-row mean and sumsq (we'll compute std here)
        idx = b * S + s
        mean = tl.load(mean_ptr + idx)
        sumsq = tl.load(sumsq_ptr + idx)
        var = sumsq - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        # Compute per-row threshold factor m
        m = mean + std * zscore

        # Iterate over K, subtract m and apply ReLU
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            mask = kk < K
            x_ptr_row = x_ptr + b * stride_b + s * stride_s + kk * stride_k
            x_vals = tl.load(x_ptr_row, mask=mask, other=0.0)
            # Subtract threshold factor and apply ReLU
            y = x_vals - m
            # ReLU: max(y, 0)
            y = tl.maximum(y, 0.0)
            out_ptr_row = out_ptr + b * stride_b + s * stride_s + kk * stride_k
            tl.store(out_ptr_row, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256, num_warps_reduce: int = 4, num_warps_element: int = 4):
        super().__init__()
        # Precompute zscore = inverse_normal_cdf(target_sparsity) on host for initialization
        # Use torch for icdf once; it's not a forward-time tensor op in host.
        self.z_score = torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(target_sparsity))
        # Triton launch params
        self.block_k = block_k
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_element = num_warps_element

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous for Triton
        if not (TRITON_AVAILABLE and x.is_cuda):
            # Fallback: pure PyTorch path (kept minimal, but note: this version uses Triton)
            # Compute mean and std per row along last dim (unbiased=False)
            x_f32 = x.to(torch.float32)
            mean = x_f32.mean(dim=-1, keepdim=True)
            std = x_f32.std(dim=-1, unbiased=False, keepdim=True)
            zscore = float(self.z_score.item())
            threshold = mean + std * zscore
            out = F.relu(x_f32 - threshold)
            return out.to(torch.bfloat16)

        # Ensure contiguous memory for simple stride-based addressing
        x = x.contiguous()

        B, S, K = x.shape
        # Allocate per-row stats (fp32) and output (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        stride_b = x.stride(0)
        stride_s = x.stride(1)
        stride_k = x.stride(2)

        # Launch reduction kernel: compute sum and sumsq per row
        grid = (B * S,)
        # Note: mean_row and sumsq_row are of size B*S; index is pid (which equals b*S + s)
        _reduce_sum_sumsq_kernel[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # Launch elementwise kernel: subtract threshold and apply ReLU
        _apply_threshold_relu_kernel[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score.item()),
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_element,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
