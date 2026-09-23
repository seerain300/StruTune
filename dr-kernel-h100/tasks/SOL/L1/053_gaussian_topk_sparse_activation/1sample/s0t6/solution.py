import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_mean_sumsq_kernel(
    x_ptr,                   # *const float, input
    mean_out_ptr,            # *float, per-row mean (length B*S)
    sumsq_out_ptr,           # *float, per-row sumsq (length B*S)
    B, S, K,                 # int32
    stride_b, stride_s, stride_k,  # int32 strides for x
    BLOCK_K: tl.constexpr,
):
    # program id: one per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # guard in case grid > B*S
    if b >= B or s >= S:
        return

    # Accumulators in fp32
    sum_ = 0.0
    sumsq_ = 0.0

    # Iterate across K in chunks
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Pointer to the row: x[b, s, offs]
        ptr = x_ptr + b * stride_b + s * stride_s + offs * stride_k
        # Load as fp32
        vals = tl.load(ptr, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        sum_ += tl.sum(vals, axis=0)
        sumsq_ += tl.sum(vals * vals, axis=0)

    Kf = tl.float32(K)
    mean = sum_ / Kf
    sumsq = sumsq_ / Kf
    # variance = E[x^2] - (E[x])^2
    var = sumsq - mean * mean
    # guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    # std = sqrt(var)
    std = tl.sqrt(var)

    # Write per-row results
    out_idx = b * S + s
    tl.store(mean_out_ptr + out_idx, mean)
    tl.store(sumsq_out_ptr + out_idx, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    x_ptr,                   # *const float, input
    out_ptr,                 # *float, output (fp32)
    mean_ptr,                # *const float, per-row mean (length B*S)
    sumsq_ptr,               # *const float, per-row sumsq (length B*S)
    B, S, K,                 # int32
    stride_b, stride_s, stride_k,   # int32 strides for x and out
    zscore,                  # float32 scalar inverse-normal at target sparsity
    BLOCK_K: tl.constexpr,
):
    # one program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    if b >= B or s >= S:
        return

    # Load per-row stats
    out_idx = b * S + s
    mean = tl.load(mean_ptr + out_idx)
    sumsq = tl.load(sumsq_ptr + out_idx)
    # Compute std from sumsq (std = sqrt(sumsq - mean^2), where sumsq was pre-averaged)
    # Here we need std = sqrt(sumsq - mean^2)
    var = sumsq - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute per-row threshold factor: mean + std * zscore
    thr = mean + std * zscore

    # Apply elementwise: out[b, s, j] = max(x[b, s, j] - thr, 0)
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_ptr_row = x_ptr + b * stride_b + s * stride_s + offs * stride_k
        x_vals = tl.load(x_ptr_row, mask=mask, other=0.0).to(tl.float32)
        out_vals = x_vals - thr
        # ReLU
        out_vals = tl.maximum(out_vals, 0.0)
        out_ptr_row = out_ptr + b * stride_b + s * stride_s + offs * stride_k
        tl.store(out_ptr_row, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256, num_warps_reduce: int = 4, num_warps_element: int = 4):
        super().__init__()
        # Precompute z_score = inverse normal cdf at target_sparsity once on host.
        # We use torch here for initialization; in forward we only pass the scalar to Triton.
        # torch.distributions.normal.icdf returns a tensor; extract float.
        try:
            from torch.distributions.normal import Normal
        except Exception:
            # Older torch may use different import; fallback path via torch.quantile if available
            # But we'll use torch.distributions.normal.icdf if available
            # If not, we can compute z via torch.erf, but simplest is to rely on torch.distributions.
            # As a safe fallback, compute z using torch.quantile if available:
            pass
        self.z_score = torch.tensor(Normal(0.0, 1.0).icdf(torch.tensor(target_sparsity)), dtype=torch.float32)
        self.block_k = block_k
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_element = num_warps_element

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: tensor of shape [B, S, K], can be float16/bfloat16/float32.
        Returns: tensor of shape [B, S, K] with bfloat16 dtype, applying adaptive threshold sparsification.
        """
        if not inputs.is_cuda:
            # CPU fallback (to maintain correctness if ever called on CPU)
            # Mirror original behavior closely: compute in float32, apply same logic, cast back to bfloat16.
            x = inputs.to(torch.float32)
            B, S, K = x.shape
            mean = x.mean(dim=-1, keepdim=True)
            # population std (unbiased=False)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            z = float(self.z_score.item())
            threshold = mean + std * z
            out = torch.relu(x - threshold)
            return out.to(torch.bfloat16)

        # Ensure contiguous input for predictable strides
        x = inputs.contiguous()
        B, S, K = x.shape

        # Allocate per-row buffers for mean and sumsq (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B * S,)
        _reduce_mean_sumsq_kernel[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            x.stride(0), x.stride(1), x.stride(2),
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # Allocate output as fp32 (we'll cast to bfloat16 at the end)
        out_fp32 = torch.empty((B, S, K), dtype=torch.float32, device=x.device)

        # Launch elementwise kernel: one program per (b, s) row
        grid = (B * S,)
        zscore = float(self.z_score.item())
        _apply_threshold_relu_kernel[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            x.stride(0), x.stride(1), x.stride(2),
            zscore,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_element,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
