import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    x_ptr,        # *const T
    out_ptr,      # *mut T
    weight_ptr,   # *const T
    B: tl.int32,  # batch size
    H: tl.int32,  # num attention heads
    L: tl.int32,  # seq_len
    D: tl.int32,  # head_dim
    stride_b: tl.int32,
    stride_h: tl.int32,
    stride_l: tl.int32,
    stride_d: tl.int32,
    eps: tl.float32,
    BLOCK: tl.constexpr,  # set BLOCK = D
):
    # One Triton program per row: (b, h, l)
    row_id = tl.program_id(0)
    l = row_id % L
    tmp = row_id // L
    h = tmp % H
    b = tmp // H

    base = b * stride_b + h * stride_h + l * stride_l

    # Compute sum of squares over the last dimension
    sum_sq = 0.0
    for idx in range(0, BLOCK):
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=(idx < D), other=0.0)
        x_f32 = x_val.to(tl.float32)
        sum_sq += x_f32 * x_f32

    mean = sum_sq / D
    inv_scale = tl.rsqrt(mean + eps)  # 1 / sqrt(mean + eps)

    # Apply RMSNorm and weight
    for idx in range(0, BLOCK):
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=(idx < D), other=0.0)
        w_val = tl.load(weight_ptr + idx, mask=(idx < D), other=0.0)
        y_f32 = (x_val.to(tl.float32) * inv_scale) * w_val.to(tl.float32)
        # Store as float32; if you need bfloat16 output, cast before store:
        # y_cast = y_f32.to(x_val.dtype)  # but x_val's dtype is not accessible here; Triton infers pointer dtype.
        tl.store(out_ptr + base + idx * stride_d, y_f32, mask=(idx < D))


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute y = weight * x / sqrt(mean(x^2) + eps) per row using Triton.
    x: [B, H, L, D], dtype bfloat16 or float32 (compute in float32).
    weight: [D], same dtype as x (compute in float32).
    Returns y with same shape/dtype as x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor for Triton kernel."
    assert x.is_contiguous(), "x must be contiguous for this kernel."
    assert weight.is_cuda, "weight must be a CUDA tensor."
    assert weight.is_contiguous(), "weight must be contiguous."

    B, H, L, D = x.shape
    # Allocate output; Triton will write float32 values; we keep output dtype as x.dtype by casting before return.
    y = torch.empty_like(x)

    # Strides for contiguous [B, H, L, D]
    stride_b = H * L * D
    stride_h = L * D
    stride_l = D
    stride_d = 1

    # One program per row (b, h, l)
    grid = (B * H * L,)

    rmsnorm_row_kernel[grid](
        x, y, weight,
        B, H, L, D,
        stride_b, stride_h, stride_l, stride_d,
        eps,
        BLOCK=D,  # process full head_dim
        num_warps=1,
        num_stages=1,
    )
    # Ensure output dtype matches input dtype (Triton kernel wrote float32; cast back)
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-ONLY implementation:
        - Compute RMSNorm for query and key using Triton.
        - Return query_norm, key_norm, key_cache (unchanged), value_cache (unchanged).
        No torch operations in host code.
        """
        # args order: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0]
        key = args[1]
        value = args[2]
        position_ids = args[3]  # unused
        key_cache = args[4]     # passed through unchanged
        value_cache = args[5]   # passed through unchanged
        cache_position = args[6]  # unused
        q_norm_weight = args[7]
        k_norm_weight = args[8]
        inv_freq = args[9]       # unused
        rms_norm_eps = args[10]  # float

        # Triton RMSNorm for query and key
        query_norm = triton_rmsnorm(query, q_norm_weight, rms_norm_eps)
        key_norm = triton_rmsnorm(key, k_norm_weight, rms_norm_eps)

        # Return the expected 4 items: (query_norm, key_norm, key_cache, value_cache)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
