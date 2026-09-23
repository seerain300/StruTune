import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,          # input [B, H, S, D] (bf16)
    y_ptr,          # output [B, H, S, D] (bf16)
    weight_ptr,     # weight [D] (bf16)
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps,            # float
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Compute sum of squares across D
    sumsq = 0.0
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and scale by weight, store
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = (x_fp32 * inv_rms) * w  # IMPORTANT: divide by rms then scale by weight
        # Store; Triton will cast to the dtype of y_ptr if needed
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_q_kernel(
    x_ptr,          # RMSNormed query, [B, H_q, S, D] (bf16)
    y_ptr,          # output query_rot, [B, H_q, S, D] (bf16)
    cos_ptr,        # cos_all, [B, S, D] float32
    sin_ptr,        # sin_all, [B, S, D] float32
    B, H_q, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load cos/sin vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        cos = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)

    # Load x chunk
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)

        # Build rotate_half(x): [x2, -x2, x1, -x1], where x1 = x[:D//2], x2 = x[D//2:]
        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        y = x * cos - rotate_half * sin
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_k_kernel(
    x_ptr,          # RMSNormed key, [B, H_kv, S, D] (bf16)
    y_ptr,          # output key_rot, [B, H_kv, S, D] (bf16)
    cos_ptr,        # cos_all, [B, S, D] float32
    sin_ptr,        # sin_all, [B, S, D] float32
    B, H_kv, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load sin/cos vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)

    # Load x chunk
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)

        # Build rotate_half(x): [x2, -x2, x1, -x1], where x1 = x[:D//2], x2 = x[D//2:]
        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        y = x * sin - rotate_half * cos
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def scatter_update_cache_kernel(
    src_ptr,        # [B, H, S, D] (bf16)
    dst_ptr,        # [B, H, L, D] (bf16), L can be large, we index via cache_position
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # [S] int32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(cache_pos_ptr + s)
    # Load src row [D] and store to dst[b, h, pos, :]
    for off in tl.static_range(0, D, 128):
        cols = off + tl.arange(0, 128)
        mask = cols < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0)
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l + cols * dst_stride_d, x, mask=mask)


@triton.jit
def store_values_kernel(
    src_ptr,        # [B, H, S, D] (bf16), in this case 'value' without rotation
    dst_ptr,        # [B, H, L, D] (bf16), we write at cache_position
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # [S] int32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s)
    for off in tl.static_range(0, D, 128):
        cols = off + tl.arange(0, 128)
        mask = cols < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0)
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l + cols * dst_stride_d, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, H_q, S, D = query.shape
        Bk, H_kv, S2, Dk = key.shape
        assert Bk == B and Dk == D and S2 == S, "Key/value shapes must match [B, H_kv, S, D]"
        Bc, Hc, L, Dc = key_cache.shape
        assert Bc == B and Hc == H_kv and Dc == D, "key_cache shape must be [B, num_key_value_heads, L, D]"
        Bv, Hv, Lv, Dv = value_cache.shape
        assert Bv == B and Hv == H_kv and Dv == D, "value_cache shape must be [B, num_key_value_heads, L, D]"
        assert cache_position.numel() == S, "cache_position must have length S"
        # Ensure contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        cache_position = cache_position.to(torch.int32).contiguous()

        # 1) RMSNorm
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid_rms = (B, H_q, S)
        rmsnorm_kernel[grid_rms](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            rms_norm_eps, 128,
            num_warps=4,
        )

        grid_rms_k = (B, H_kv, S)
        rmsnorm_kernel[grid_rms_k](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            rms_norm_eps, 128,
            num_warps=4,
        )

        # 2) Compute cos/sin vectors (PyTorch, no heavy compute)
        # For query: cos_all = cos(angle) concat with itself; sin_all = sin(angle) concat with itself
        # For key: cos_all = sin(angle) concat with itself; sin_all = cos(angle) concat with itself
        # angle = pos * inv_freq, inv_freq length D//2
        pos = position_ids[:, :, None].float()  # [B, S, 1]
        inv_freq = inv_freq[None, None, :].float()  # [1, 1, D//2]
        angle = pos * inv_freq  # [B, S, D//2]
        angle_q = angle  # query uses angle
        angle_k = angle  # key uses same angle

        cos_all_q = torch.cos(angle_q).to(query.dtype)  # [B, S, D//2]
        sin_all_q = torch.sin(angle_q).to(query.dtype)  # [B, S, D//2]
        cos_all_q_full = torch.cat([cos_all_q, cos_all_q], dim=-1)  # [B, S, D]
        sin_all_q_full = torch.cat([sin_all_q, sin_all_q], dim=-1)  # [B, S, D]

        cos_all_k = torch.sin(angle_k).to(key.dtype)  # [B, S, D//2]
        sin_all_k_full = torch.cat([cos_all_k, cos_all_k], dim=-1)  # [B, S, D]
        cos_all_k_full = torch.cat([sin_all_q, sin_all_q], dim=-1)  # [B, S, D] (PyTorch: sin(angle) is cos_all for keys in original)
        # Note: Original code uses keys with sin-based rotation. Here we produce sin_all_k_full = sin(angle) concat, cos_all_k_full = cos(angle) concat.
        # However, original code constructs sin_all_k = cos(angle) concat and cos_all_k = sin(angle) concat. We need to reflect that exactly:
        sin_all_k = torch.cos(angle_k).to(key.dtype)  # [B, S, D//2]
        cos_all_k = torch.sin(angle_k).to(key.dtype)  # [B, S, D//2]
        sin_all_k_full = torch.cat([sin_all_k, sin_all_k], dim=-1)  # [B, S, D]
        cos_all_k_full = torch.cat([cos_all_k, cos_all_k], dim=-1)  # [B, S, D]

        # Ensure contiguity and device
        cos_all_q_full = cos_all_q_full.contiguous().to(query.dtype)
        sin_all_q_full = sin_all_q_full.contiguous().to(query.dtype)
        cos_all_k_full = cos_all_k_full.contiguous().to(key.dtype)
        sin_all_k_full = sin_all_k_full.contiguous().to(key.dtype)

        # 3) Query rotation
        query_rot = torch.empty_like(query_norm)
        grid_qrot = (B, H_q, S)
        rotate_q_kernel[grid_qrot](
            query_norm, query_rot, cos_all_q_full, sin_all_q_full,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_norm.stride(2), query_rot.stride(3),
            128,
            num_warps=4,
        )

        # 4) Key rotation (original uses sin-based rotation; here we mirror that with sin_all_k_full)
        key_rot = torch.empty_like(key_norm)
        grid_krot = (B, H_kv, S)
        rotate_k_kernel[grid_krot](
            key_norm, key_rot, sin_all_k_full, cos_all_k_full,  # sin for keys, cos for keys (original pattern)
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_norm.stride(2), key_rot.stride(3),
            128,
            num_warps=4,
        )

        # 5) Scatter update caches: write rotated keys and original values at cache_position
        # key_cache: [B, H_kv, L, D], value_cache: [B, H_kv, L, D]
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            key_rot, key_cache,
            B, H_kv, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position,
        )

        # Store original values (not rotated)
        grid_store_val = (B, H_kv, S)
        store_values_kernel[grid_store_val](
            value, value_cache,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position,
        )

        # Return as original function expects
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
