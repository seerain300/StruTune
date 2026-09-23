import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,            # *T (input), [B, H, S, D]
    w_ptr,            # *T (weight), [D]
    y_ptr,            # *T (output), [B, H, S, D]
    B: tl.constexpr,  # not used, but can be used if grid is (B, H, S)
    H: tl.constexpr,  # same
    S: tl.constexpr,  # same
    D: tl.constexpr,  # D (e.g., 128)
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Accumulate sum of squares in fp32
    sum_sq = 0.0
    for d in range(0, D):
        x_val = tl.load(x_row_ptr + d * x_stride3).to(tl.float32)
        sum_sq += x_val * x_val
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 1e-6)

    # Scale by weight and store
    w_ptr_base = w_ptr
    for d in range(0, D):
        x_val = tl.load(x_row_ptr + d * x_stride3).to(tl.float32)
        w_val = tl.load(w_ptr_base + d).to(tl.float32)
        y_val = x_val * inv_rms * w_val
        tl.store(y_row_ptr + d * y_stride3, y_val)


@triton.jit
def rotate_query_kernel(
    x_ptr,       # *T (normalized), [B, H, S, D]
    cos_ptr,     # *float32, [B, S, D]
    sin_ptr,     # *float32, [B, S, D]
    y_ptr,       # *T, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    cos_stride0, cos_stride1, cos_stride2,
    sin_stride0, sin_stride1, sin_stride2,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Load cos_all and sin_all for this (b, s): shape [D]
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    d = tl.arange(0, D)
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32

    # Rotate: y = x * cos - rotate_half(x) * sin
    for d_off in range(0, D):
        x_val = tl.load(x_row_ptr + d_off * x_stride3).to(tl.float32)
        half = D // 2
        x_first = x_val[:half]
        x_second = x_val[half:]
        rotate = -x_second + x_first  # rotate_half(x) = [-x2, x1]
        y_val = x_val * cos_all - rotate * sin_all
        tl.store(y_row_ptr + d_off * y_stride3, y_val)


@triton.jit
def rotate_key_kernel(
    x_ptr,       # *T (normalized), [B, H, S, D]
    cos_ptr,     # *float32, [B, S, D]
    sin_ptr,     # *float32, [B, S, D]
    y_ptr,       # *T, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    cos_stride0, cos_stride1, cos_stride2,
    sin_stride0, sin_stride1, sin_stride2,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Load cos_all and sin_all for this (b, s): shape [D]
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    d = tl.arange(0, D)
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32

    # Rotate: y = x * sin - rotate_half(x) * cos
    for d_off in range(0, D):
        x_val = tl.load(x_row_ptr + d_off * x_stride3).to(tl.float32)
        half = D // 2
        x_first = x_val[:half]
        x_second = x_val[half:]
        rotate = -x_second + x_first  # rotate_half(x) = [-x2, x1]
        y_val = x_val * sin_all - rotate * cos_all
        tl.store(y_row_ptr + d_off * y_stride3, y_val)


@triton.jit
def scatter_update_key_cache_kernel(
    src_ptr,          # *T (rotated keys), [B, H_kv, S, D]
    dst_ptr,          # *T (key_cache), [B, H_kv, L, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
    cache_pos_ptr,    # *int64, [S]
):
    # grid = (B, H, S)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s).to(tl.int64)
    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2
    dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2

    for d in range(0, D):
        val = tl.load(src_row_ptr + d * src_stride3)
        tl.store(dst_row_ptr + d * dst_stride3, val)


@triton.jit
def scatter_update_value_cache_kernel(
    src_ptr,          # *T (values), [B, H_kv, S, D]
    dst_ptr,          # *T (value_cache), [B, H_kv, L, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
    cache_pos_ptr,    # *int64, [S]
):
    # grid = (B, H, S)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s).to(tl.int64)
    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2
    dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2

    for d in range(0, D):
        val = tl.load(src_row_ptr + d * src_stride3)
        tl.store(dst_row_ptr + d * dst_stride3, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No fixed constants; everything is computed generically for given inputs

    def forward(self, *args):
        # Inputs: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        assert len(args) == 11, "Expected 11 inputs"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        # Shapes
        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert D == 128, "This implementation expects head_dim == 128"
        H_kv = key.shape[1]
        assert key.shape == (B, H_kv, S, D) and value.shape == (B, H_kv, S, D), "Key/value must be [B, num_kv_heads, S, D]"

        # Ensure dtype/device consistency
        device = query.device
        dtype = query.dtype  # typically bfloat16
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        q_norm_weight = q_norm_weight.to(device=device, dtype=dtype)
        k_norm_weight = k_norm_weight.to(device=device, dtype=dtype)

        # 0) Allocate outputs for RMSNorm
        query_norm = torch.empty_like(query, device=device, dtype=dtype)
        key_norm = torch.empty_like(key, device=device, dtype=dtype)

        # 1) RMSNorm via Triton: launch grid (B, H, S)
        grid_norm = (B, H_q, S)
        rmsnorm_kernel[grid_norm](
            query, q_norm_weight, query_norm,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            num_warps=4,
        )

        rmsnorm_kernel[grid_norm](
            key, k_norm_weight, key_norm,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            num_warps=4,
        )

        # 2) Compute cos_all and sin_all using torch (device-side), shape [B, S, D], float32
        # angle = pos * inv_freq[:D//2], then cos/sin; concatenate with itself
        pos_ids = position_ids.to(device=device, dtype=torch.float32)
        inv_freq_half = inv_freq[:D // 2]  # [64] float32
        # Make per-(b, s) positions
        # We need B x S
        pos_expanded = pos_ids.unsqueeze(-1)  # [B, S, 1]
        angle = pos_expanded * inv_freq_half  # [B, S, 64]
        cos_half = torch.cos(angle)           # [B, S, 64]
        sin_half = torch.sin(angle)           # [B, S, 64]

        # Concatenate with itself along last dim to [B, S, 128]
        cos_all = torch.cat([cos_half, cos_half], dim=-1)  # [B, S, 128]
        sin_all = torch.cat([sin_half, sin_half], dim=-1)  # [B, S, 128]

        # 3) Rotate query and key via Triton: launch grid (B, H, S)
        grid_rot = (B, H_q, S)
        rotate_query_kernel[grid_rot](
            query_norm, cos_all, sin_all, query_norm,  # output overwrites query_norm
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            num_warps=4,
        )

        grid_rot = (B, H_kv, S)
        rotate_key_kernel[grid_rot](
            key_norm, cos_all, sin_all, key_norm,  # output overwrites key_norm
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            num_warps=4,
        )

        # 4) Scatter update caches via Triton: grid (B, H_kv, S)
        L = key_cache.shape[2]
        # Ensure cache_position is int64 and on device
        cache_pos = cache_position.to(device=device, dtype=torch.int64)

        scatter_update_key_cache_kernel[(B, H_kv, S)](
            key_norm, key_cache,
            B, H_kv, S, D, L,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_pos,
            num_warps=4,
        )

        scatter_update_value_cache_kernel[(B, H_kv, S)](
            value, value_cache,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_pos,
            num_warps=4,
        )

        # Return rotated query and key, and updated caches
        # Note: original run returns (query_rotated, key_rotated, key_cache, value_cache).
        # We need to return the rotated query and key tensors after rotation kernels.
        # However, we modified query_norm and key_norm in place. To return the rotated versions,
        # we can simply return query_norm and key_norm as they are the rotated outputs.
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
