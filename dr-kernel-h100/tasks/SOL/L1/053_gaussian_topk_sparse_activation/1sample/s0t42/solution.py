import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_kernel_1d(
    x_ptr,            # *T (input tensor)
    mean_ptr,         # *fp32 (per-row mean, size B*S)
    sumsq_ptr,        # *fp32 (per-row sumsq, size B*S)
    B: tl.constexpr,  # batch size
    S: tl.constexpr,  # seq_len
    K,                # intermediate_size (int)
    stride_b,         # stride for batch (in elements)
    stride_s,         # stride for seq (in elements)
    stride_k,         # stride for features (in elements)
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(axis=0)  # row index in [0, B*S)
    b = r // S
    s = r % S
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # Loop over K in tiles
    offs = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        kk = k + offs
        mask = kk < K
        ptrs = x_ptr + base + kk * stride_k
        x = tl.load(ptrs, mask=mask, other=0.0)  # load as original dtype
        x_f32 = x.to(tl.float32)
        # Sum over tile
        sum_x += tl.sum(x_f32, axis=0)
        sum_x2 += tl.sum(x_f32 * x_f32, axis=0)

    # Compute mean and sumsq (population moments over K)
    mean = sum_x / K
    sumsq = sum_x2 / K

    # Store per-row results
    tl.store(mean_ptr + r, mean)
    tl.store(sumsq_ptr + r, sumsq)


@triton.jit
def _apply_threshold_relu_kernel_1d(
    x_ptr,            # *T (input tensor)
    out_ptr,          # *fp32 (output buffer, size B*S*K)
    mean_ptr,         # *fp32 (per-row mean, size B*S)
    sumsq_ptr,        # *fp32 (per-row sumsq, size B*S)
    B: tl.constexpr,  # batch size
    S: tl.constexpr,  # seq_len
    K,                # intermediate_size (int)
    stride_b,         # stride for batch (in elements)
    stride_s,         # stride for seq (in elements)
    stride_k,         # stride for features (in elements)
    z_score,          # scalar float
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(axis=0)  # row index in [0, B*S)
    b = r // S
    s = r % S
    base = b * stride_b + s * stride_s

    # Load per-row statistics
    mean = tl.load(mean_ptr + r)
    sumsq = tl.load(sumsq_ptr + r)
    std = tl.sqrt(max(sumsq - mean * mean, 0.0))  # population std

    threshold = mean + std * z_score

    # Compute output for this row
    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        mask = kk < K
        x = tl.load(x_ptr + base + kk * stride_k, mask=mask, other=0.0)
        y = x.to(tl.float32) - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        out_ptrs = out_ptr + r * K + kk
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity)
        # For target_sparsity=0.9, z ≈ 1.2815515655446004
        # We implement a simple wrapper using torch for correctness, but we will
        # not use torch in forward. z_score is passed to Triton kernels as a scalar.
        # Compute using torch only for initialization:
        # Using torch.special.erf inverse approximation (torch.special.erfinv) if available:
        try:
            from torch.special import erfinv
            z_score = (1.0 - target_sparsity) * erfinv(1.0 - 2.0 * target_sparsity)
        except Exception:
            # Fallback value for 90% quantile in standard normal
            z_score = 1.2815515655446004
        self.z_score = float(z_score)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, K], dtype likely bfloat16 or float32
        # We will perform computations in fp32 in Triton kernels.
        B, S, K = x.shape
        # Make input contiguous so stride_k == 1 and addressing is simple
        x = x.contiguous()

        # Prepare strides (in elements)
        stride_b, stride_s, stride_k = x.stride()  # for a contiguous tensor: (S*K, K, 1)

        # Per-row buffers (fp32) of shape [B*S]
        total_rows = B * S
        mean_row = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per row
        grid = (total_rows,)
        _reduce_sum_sumsq_kernel_1d[grid](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) of size B*S*K
        out_fp32 = torch.empty(total_rows * K, dtype=torch.float32, device=x.device)

        # Launch elementwise kernel: one program per row
        _apply_threshold_relu_kernel_1d[grid](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
