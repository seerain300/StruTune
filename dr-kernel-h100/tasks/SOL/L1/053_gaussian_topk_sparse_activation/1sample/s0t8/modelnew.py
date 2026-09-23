import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel(
    x_ptr,          # *const float32
    mean_ptr,       # *float32, size B*S
    sumsq_ptr,      # *float32, size B*S
    B, S, K,        # int32
    stride_b, stride_s, stride_k,  # int32
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # one program per (b, s) row
    # We launch with grid=(B*S,)
    # compute b, s from pid
    s = pid % S
    b = pid // S
    base = b * stride_b + s * stride_s

    sum_ = 0.0
    sumsq_ = 0.0

    # loop over K in tiles
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        # accumulate sums
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)

    mean = sum_ / K
    sumsq = sumsq_ / K
    var = sumsq - mean * mean
    # var should be non-negative; sqrt handles negative due to fp rounding gracefully
    std = tl.sqrt(var)

    # write per-row mean and sumsq
    idx = pid  # pid equals b*S + s
    tl.store(mean_ptr + idx, mean)
    tl.store(sumsq_ptr + idx, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,          # *const float32
    out_ptr,        # *float32
    mean_ptr,       # *const float32, size B*S
    sumsq_ptr,      # *const float32, size B*S
    B, S, K,        # int32
    stride_b, stride_s, stride_k,  # int32
    zscore,         # float32
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S
    b = pid // S
    base = b * stride_b + s * stride_s

    # load per-row stats
    idx = pid
    mean = tl.load(mean_ptr + idx)
    sumsq = tl.load(sumsq_ptr + idx)
    std = tl.sqrt(sumsq - mean * mean)
    # compute threshold factor: mean + std * zscore
    m = mean + std * zscore

    # subtract m from each element and apply ReLU
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        y = x - m  # m is scalar; broadcasts over vector
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs * stride_k, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute zscore (inverse normal CDF) for target_sparsity once.
        # Use torch for initialization; not used in forward on tensors.
        self.z_score = torch.tensor(torch.distributions.normal._percentile_to_value(torch.tensor(target_sparsity)), dtype=torch.float32)
        # Tunable kernel params
        self.block_k = 256
        self.num_warps_reduce = 4
        self.num_warps_element = 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure float32 inputs for numeric stability; original code casts anyway.
        if inputs.dtype != torch.float32:
            inputs = inputs.to(torch.float32)
        x = inputs.contiguous()  # ensure simple strides for Triton

        B, S, K = x.shape
        stride_b = x.stride(0)
        stride_s = x.stride(1)
        stride_k = x.stride(2)

        # Allocate per-row buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: compute sum and sumsq per row
        grid = (B * S,)
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
        out_fp32 = torch.empty_like(x, dtype=torch.float32)
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