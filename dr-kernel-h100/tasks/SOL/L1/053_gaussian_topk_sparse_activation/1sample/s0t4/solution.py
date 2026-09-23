import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_mean_sumsq_kernel(
    x_ptr,
    mean_ptr,
    sumsq_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map linear pid to (b, s)
    b = pid // S
    s = pid % S

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over K in chunks
    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Compute element offsets
        ptrs = x_ptr + b * stride_b + s * stride_s + offs * stride_k
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Compute mean and sum of squares per row (population stats)
    mean = sum_val / K
    # population variance: E[x^2] - (E[x])^2
    var = sumsq_val / K - mean * mean
    # ensure non-negative due to floating-point rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results to per-row buffers
    mean_ptr[pid] = mean
    sumsq_ptr[pid] = sumsq_val  # we'll derive std on host; here we store sumsq to keep interface consistent
    # Note: sumsq_ptr[pid] will be used to derive std in host by computing std = sqrt(sumsq/K - mean^2).


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    std_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    z_score,  # scalar float32
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Load per-row mean and std (fp32)
    mean = mean_ptr[pid]
    sumsq = std_ptr[pid]  # actually sumsq row; we need to compute std here
    # Compute std per row: std = sqrt(sumsq/K - mean^2)
    var = sumsq / K - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Precompute threshold per feature: mean + std * z_score
    thresh = mean + std * z_score

    # Iterate over K and apply: out = max(x - thresh, 0)
    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K

        in_ptrs = x_ptr + b * stride_b + s * stride_s + offs * stride_k
        in_vals = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Broadcast threshold (scalar) to vector
        out_vals = in_vals - thresh
        # ReLU
        out_vals = tl.maximum(out_vals, 0.0)

        out_ptrs = out_ptr + b * stride_b + s * stride_s + offs * stride_k
        tl.store(out_ptrs, out_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute inverse normal CDF for sparsity = 0.9 on host once.
        # This avoids any torch ops in the forward host path.
        # torch.distributions.normal.icdf returns the value p such that Phi(p) = target_sparsity.
        # We want z = icdf(target_sparsity), i.e., z with P(X <= z) = target_sparsity.
        self.z_score = float(torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(target_sparsity, dtype=torch.float32)))
        self.block_k = block_k

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure we run on CUDA and use Triton kernels.
        if not inputs.is_cuda:
            # Fallback to original PyTorch logic if not on CUDA.
            # This maintains correctness in non-CUDA environments.
            inputs_f32 = inputs.to(torch.float32)
            inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
            inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
            target_sparsity_tensor = torch.tensor(self.z_score, dtype=torch.float32, device=inputs.device)
            std_multiplier = torch.distributions.normal.Normal(0, 1).icdf(target_sparsity_tensor)  # just a scalar
            cutoff_threshold = inputs_mean + inputs_std * std_multiplier
            sparse_output = torch.relu(inputs_f32 - cutoff_threshold)
            return sparse_output.to(torch.bfloat16)

        # Input: [B, S, K]
        x = inputs.contiguous()
        B, S, K = x.shape
        # Allocate per-row buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Strides in elements
        stride_b, stride_s, stride_k = x.stride()

        # 1) Reduction: compute per-row mean and sum of squares
        grid_reduce = (B * S,)
        _reduce_mean_sumsq_kernel[grid_reduce](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=1,
        )

        # 2) Elementwise: apply threshold = mean + std * z_score and ReLU
        out_fp32 = torch.empty_like(x, dtype=torch.float32, device=x.device)
        grid_apply = (B * S,)
        _apply_threshold_relu_kernel[grid_apply](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score),  # scalar, fp32
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=1,
        )

        # Match original return dtype: bfloat16
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
