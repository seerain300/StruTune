import torch
import triton
import triton.language as tl


# RMSNorm: one program per (b, h, s) row, reduce across D, compute inv_rms, scale by weight, store fp32
@triton.jit
def rms_norm_kernel(
    X_ptr, Y_ptr, Weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Row base pointers
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    y_row_ptr = Y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s

    sum_sq = 0.0
    # Reduce across D in chunks of BLOCK_SIZE
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + eps)  # 1 / sqrt(mean + eps)

    # Scale by weight
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(y_row_ptr + idx * y_stride_d, y, mask=mask)


# Rotate query: y = x * cos_all - rotate_half(x) * sin_all
# Compute cos_all and sin_all inside kernel (concatenate sin(pos * inv_freq) with itself).
@triton.jit
def rotate_query_kernel(
    X_ptr, Out_ptr, Position_ptr, InvFreq_ptr,
    B, H, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load position id
    pos = tl.load(Position_ptr + b * S + s)  # int32 scalar

    # Build cos_all and sin_all (length D), using inv_freq of length HALF_D
    cos_all = tl.zeros([D], dtype=tl.float32)
    sin_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle = pos * tl.load(InvFreq_ptr + j)  # scalar
        c = tl.cos(angle)
        s_ = tl.sin(angle)
        # even indices get cos, odd indices get sin
        even_mask = (tl.arange(0, D) % 2) == 0
        cos_all = tl.where(even_mask, cos_all + c, cos_all + s_)
        sin_all = tl.where((tl.arange(0, D) % 2) == 1, sin_all + s_, sin_all + c)

    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        # rotate_half(x): [-x[D//2:], x[:D//2]]
        half = D // 2
        x_half = x[half:]
        x_first = x[:half]
        x_rot = tl.concatenate([-x_half, x_first], axis=0)
        y = x * cos_all[offs : offs + BLOCK_SIZE] - x_rot * sin_all[offs : offs + BLOCK_SIZE]
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Rotate key: y = x * sin_all - rotate_half(x) * cos_all
@triton.jit
def rotate_key_kernel(
    X_ptr, Out_ptr, Position_ptr, InvFreq_ptr,
    B, H, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(Position_ptr + b * S + s)  # int32 scalar

    # Build sin_all and cos_all (length D)
    sin_all = tl.zeros([D], dtype=tl.float32)
    cos_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle = pos * tl.load(InvFreq_ptr + j)
        s_ = tl.sin(angle)
        c = tl.cos(angle)
        even_mask = (tl.arange(0, D) % 2) == 0
        sin_all = tl.where(even_mask, sin_all + s_, sin_all + s_)  # even -> sin
        cos_all = tl.where(even_mask, cos_all + c, cos_all + c)    # even -> cos, odd -> cos (as per original)
        # Correction: original code uses sin for key rotation. So odd indices also get sin.
        odd_mask = (tl.arange(0, D) % 2) == 1
        sin_all = tl.where(odd_mask, sin_all + s_, sin_all + s_)
        cos_all = tl.where(odd_mask, cos_all + c, cos_all + c)

    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_half = x[half:]
        x_first = x[:half]
        x_rot = tl.concatenate([-x_half, x_first], axis=0)
        y = x * sin_all[offs : offs + BLOCK_SIZE] - x_rot * cos_all[offs : offs + BLOCK_SIZE]
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Scatter update caches: For each (b,h,s), write to key_cache[b, h, cache_position[s], :] and value_cache
@triton.jit
def scatter_update_cache_kernel(
    Src_ptr, Out_ptr,
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    out_stride_b, out_stride_h, out_stride_l, out_stride_d,
    positions_ptr,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(positions_ptr + b * S + s)  # int32 scalar
    src_row_ptr = Src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + pos * out_stride_l

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(src_row_ptr + idx * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row_ptr + idx * out_stride_d, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq
        (query, key, value, position_ids, key_cache, value_cache, cache_position,
         q_norm_weight, k_norm_weight, inv_freq) = args

        device = query.device
        dtype = query.dtype  # typically bfloat16

        B, H_q, S, D = query.shape
        H_key = key.shape[1]
        assert H_key == H_q // 2, "num_kv_heads must be half of num_q_heads."

        # 1) RMSNorm: compute in fp32
        query_norm = torch.empty(query.shape, device=device, dtype=torch.float32)
        key_norm = torch.empty(key.shape, device=device, dtype=torch.float32)

        rms_norm_kernel[(B, H_q, S)](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        rms_norm_kernel[(B, H_key, S)](
            key, key_norm, k_norm_weight,
            B, H_key, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # 2) Rotate query and key using Triton (compute cos/sin inside kernels)
        position_ids_i32 = position_ids.to(torch.int32)

        query_rot = torch.empty(query.shape, device=device, dtype=torch.float32)
        rotate_query_kernel[(B, H_q, S)](
            query_norm, query_rot, position_ids_i32, inv_freq,
            B, H_q, S, D, D // 2,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        key_rot = torch.empty(key.shape, device=device, dtype=torch.float32)
        rotate_key_kernel[(B, H_key, S)](
            key_norm, key_rot, position_ids_i32, inv_freq,
            B, H_key, S, D, D // 2,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # 3) Scatter update caches using Triton
        cache_position_i32 = cache_position.to(torch.int32)

        # key_cache update: write rotated keys at cache_position
        scatter_update_cache_kernel[(B, H_key, S)](
            key_rot, key_cache,
            B, H_key, S, D,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position_i32,
            BLOCK_SIZE=128,
        )

        # value_cache update: write original values at cache_position
        scatter_update_cache_kernel[(B, H_key, S)](
            value, value_cache,
            B, H_key, S, D,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position_i32,
            BLOCK_SIZE=128,
        )

        # Return the rotated query and key, and updated caches (the original run returns three things)
        # The original forward returns (query_rotated, key_rotated, key_cache, value_cache).
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
