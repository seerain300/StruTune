import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_mean_sumsq_kernel(
    x_ptr,            # *float32
    mean_ptr,         # *float32, length B*S
    sumsq_ptr,        # *float32, length B*S
    B, S, K,          # int32
    stride_b, stride_s, stride_k,  # int32 (strides for x)
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*S-1
    b = pid // S
    s = pid % S

    # Compute base offset for this (b, s) row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for start in range(0, K, BLOCK_K):
        kk = start + tl.arange(0, BLOCK_K)
        mask = kk < K
        offs = base + kk * stride_k
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)  # fp32
        acc_sum += tl.sum(x_vals, axis=0)
        acc_sumsq += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and sumsq (already divided by K)
    Kf = tl.full((), K, tl.float32)
    mean = acc_sum / Kf
    sumsq = acc_sumsq / Kf

    # Write results for this row
    out_idx = pid
    tl.store(mean_ptr + out_idx, mean)
    tl.store(sumsq_ptr + out_idx, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,             # *float32 input
    out_ptr,           # *float32 output
    mean_ptr,          # *float32, length B*S
    sumsq_ptr,         # *float32, length B*S
    B, S, K,           # int32
    stride_b, stride_s, stride_k,  # int32 (strides for x/out)
    zscore,            # float32 scalar
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*S-1
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    # Load per-row mean and sumsq
    out_idx = pid
    mean = tl.load(mean_ptr + out_idx)
    sumsq = tl.load(sumsq_ptr + out_idx)

    # Compute std: sqrt(max(var, 0))
    var = sumsq - mean * mean
    # Guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute threshold for this row: mean + std * zscore
    thr = mean + std * zscore

    # Elementwise: y = max(x - thr, 0) across K
    for start in range(0, K, BLOCK_K):
        kk = start + tl.arange(0, BLOCK_K)
        mask = kk < K
        offs = base + kk * stride_k
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)  # fp32
        y = x_vals - thr
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256,
                 num_warps_reduce: int = 4, num_warps_element: int = 4):
        super().__init__()
        # Precompute zscore for target_sparsity on host (torch init only)
        # Use a robust way: normal icdf at sparsity
        # Note: this is a host-side computation; forward uses only a scalar.
        import torch.distributions as dist
        zscore = float(dist.normal.Normal(0, 1).icdf(1.0 - target_sparsity))  # sparsity is 1 - quantile
        self.z_score = torch.tensor(zscore, dtype=torch.float32)

        # Triton tiling parameters
        self.block_k = block_k
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_element = num_warps_element

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous for Triton
        assert inputs.is_cuda, "ModelNew.forward expects a CUDA tensor"
        x = inputs.contiguous()
        assert x.dtype == torch.bfloat16, "This implementation expects bfloat16 input to match original"
        # Convert to fp32 for numerical stability
        x = x.to(torch.float32)

        B, S, K = x.shape
        # Prepare output (fp32) and per-row stats buffers
        out_fp32 = torch.empty_like(x)  # fp32 output from Triton
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        stride_b = x.stride(0)
        stride_s = x.stride(1)
        stride_k = x.stride(2)

        # Launch reduction kernel: compute mean and sumsq per row
        grid = (B * S,)
        _reduce_mean_sumsq_kernel[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=1,
        )

        # Launch elementwise kernel: subtract threshold and ReLU
        grid = (B * S,)
        zscore = float(self.z_score.item())
        _apply_threshold_relu_kernel[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            zscore,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_element,
            num_stages=1,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)