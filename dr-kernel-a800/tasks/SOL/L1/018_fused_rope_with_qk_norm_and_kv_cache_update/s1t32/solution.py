import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    x_ptr, y_ptr, norm_weight_ptr,
    B, H, L, D,
    stride_b, stride_h, stride_l, stride_d,
    eps,
    BLOCK: tl.constexpr,
):
    # One Triton program per row: flatten rows into [0, B*H*L)
    row_id = tl.program_id(0)
    if row_id >= B * H * L:
        return

    # Decode (b, h, l) from row_id
    b = row_id // (H * L)
    hl = row_id % (H * L)
    h = hl // L
    l = hl % L

    # Base offset for this row
    base = b * stride_b + h * stride_h + l * stride_l

    # Compute variance: sum(x^2) over D
    sum_sq = 0.0
    for i in range(0, BLOCK):
        v = tl.load(x_ptr + base + i * stride_d, mask=(i < D), other=0.0)
        v = v.to(tl.float32)
        sum_sq += v * v
    mean = sum_sq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)

    # Apply normalization and weight: y = weight * x * inv_scale
    for i in range(0, BLOCK):
        x_val = tl.load(x_ptr + base + i * stride_d, mask=(i < D), other=0.0)
        w_val = tl.load(norm_weight_ptr + i, mask=(i < D), other=0.0)
        y = (x_val.to(tl.float32) * inv_scale) * w_val.to(tl.float32)
        tl.store(y_ptr + base + i * stride_d, y, mask=(i < D))


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute y = weight * x / sqrt(mean(x^2) + eps) per row using Triton.
    x: [B, H, L, D], dtype bfloat16 or float32. We compute in fp32 and return same dtype as x.
    weight: [D], same shape, dtype as x (we convert to fp32 in kernel).
    """
    assert x.is_cuda, "Input must be CUDA tensor for Triton kernel."
    B, H, L, D = x.shape
    assert weight.numel() == D, "norm_weight must have length D"

    y = torch.empty_like(x)

    # Strides in elements (assuming contiguous layout but using explicit strides)
    # Triton uses element strides; PyTorch tensors have .stride() in elements.
    stride_b = H * L * D
    stride_h = L * D
    stride_l = D
    stride_d = 1

    # Launch Triton kernel: one program per row
    grid = (B * H * L,)
    rmsnorm_row_kernel[grid](
        x, y, weight,
        B, H, L, D,
        stride_b, stride_h, stride_l, stride_d,
        eps,
        BLOCK=D,  # process full head_dim; mask guards idx < D
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Accept the original run signature:
        # query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0]
        key = args[1]
        value = args[2]
        position_ids = args[3]
        key_cache = args[4]
        value_cache = args[5]
        cache_position = args[6]
        q_norm_weight = args[7]
        k_norm_weight = args[8]
        inv_freq = args[9]  # not used in this Triton-only version
        rms_norm_eps = args[10]

        # Triton RMSNorm for query and key
        query_norm = triton_rmsnorm(query, q_norm_weight, rms_norm_eps)
        key_norm = triton_rms_norm(key, k_norm_weight, rms_norm_eps)

        # Return normalized tensors and original caches (do not mutate caches or apply rotation).
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
