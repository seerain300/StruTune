import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    x_stride0, x_stride1, x_stride2,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S
    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid * sum_stride0, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, y_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     x_stride0, x_stride1, x_stride2,
                     y_stride0, y_stride1, y_stride2,
                     w_stride0, sum_stride0,
                     eps: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_sums_ptr + pid * sum_stride0).to(tl.float32)
    H_f = H.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_val / (H_f * H_f) + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + w_stride0 + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        offs_y = b * y_stride0 + head * y_stride1 + s * y_stride2 + idx
        tl.store(y_ptr + offs_y, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                               B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                               H2: tl.constexpr,
                               position_stride0,
                               cos_stride0, cos_stride1, cos_stride2,
                               sin_stride0, sin_stride1, sin_stride2,
                               BLOCK_H: tl.constexpr):
    # grid = (B, S)
    b = tl.program_id(0)
    s = tl.program_id(1)
    # Position
    pos = tl.load(position_ids_ptr + b * position_stride0 + s)
    # Build emb vector of length H: duplicate first half
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half = idx < H2
        # Load inv_freq for first half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate second half
        emb = tl.where(first_half, emb_first, emb_first)
        c = tl.cos(emb).to(tl.float32)
        s_ = tl.sin(emb).to(tl.float32)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           x_stride0, x_stride1, x_stride2,
                           out_stride0, out_stride1, out_stride2,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # grid = (B * num_heads * S,)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # Load first half from x[half + idx], negate; load second half from x[idx - half]
        for i in range(0, BLOCK_H):
            if (off + i) < half:
                rotated[off + i] = -tl.load(x_ptr + (b * x_stride0 + head * x_stride1 + s * x_stride2 + (half + (off + i))), mask=True, other=0.0).to(tl.float32)
            else:
                rotated[off + i] = tl.load(x_ptr + (b * x_stride0 + head * x_stride1 + s * x_stride2 + ((off + i) - half)), mask=True, other=0.0).to(tl.float32)

        y_vals = x_vals * cos_vals + rotated * sin_vals
        offs_out = b * out_stride0 + head * out_stride1 + s * out_stride2 + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         key_stride0, key_stride1, key_stride2,
                         value_stride0, value_stride1, value_stride2,
                         key_cache_stride0, key_cache_stride1, key_cache_stride2,
                         value_cache_stride0, value_cache_stride1, value_cache_stride2,
                         cache_stride0,
                         BLOCK_H: tl.constexpr):
    # grid = (B * num_kv_heads * S,)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv = (pid % (num_kv_heads * S)) // S
    s = pid % S
    dest_pos = tl.load(cache_pos_ptr + s * cache_stride0).to(tl.int32)

    base_k = b * key_stride0 + kv * key_stride1 + s * key_stride2
    base_v = b * value_stride0 + kv * value_stride1 + s * value_stride2

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        xk = tl.load(key_ptr + base_k + idx, mask=mask, other=0.0).to(tl.float32)
        xv = tl.load(value_ptr + base_v + idx, mask=mask, other=0.0).to(tl.float32)

        base_ck = b * key_cache_stride0 + kv * key_cache_stride1 + dest_pos * key_cache_stride2
        base_cv = b * value_cache_stride0 + kv * value_cache_stride1 + dest_pos * value_cache_stride2

        tl.store(key_cache_ptr + base_ck + idx, xk, mask=mask)
        tl.store(value_cache_ptr + base_cv + idx, xv, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters; kernels handle all math

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        _, num_kv_heads, _, _ = key.shape
        assert H == key.shape[-1] == value.shape[-1], "head_dim mismatch"
        assert position_ids.shape == (B, S)
        assert key_cache.shape == (B, num_kv_heads, key_cache.shape[2], H)
        assert value_cache.shape == (B, num_kv_heads, value_cache.shape[2], H)
        assert cache_position.shape == (S,)
        # Host-only setup: allocate outputs and constants
        # 1) RMSNorm sum for query
        sum_query = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        BLOCK_H = 128
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](
            query, sum_query,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_query.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 2) RMSNorm for query -> query_norm (float32 compute, output bfloat16)
        query_norm = torch.empty_like(query, dtype=torch.float32)
        grid_rms_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_rms_q](
            query, q_norm_weight, query_norm, sum_query,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            k_norm_weight.stride(0), sum_query.stride(0),
            rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4
        )

        # 3) Rotate sin/cos for each (b, s) -> [B, S, H] float32
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](
            position_ids, inv_freq, cos, sin,
            B, S, H, H2,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 4) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty_like(query_norm, dtype=torch.bfloat16)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 5) RMSNorm sum for key
        sum_key = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](
            key, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_key.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 6) RMSNorm for key -> key_norm (float32 compute, output bfloat16)
        key_norm = torch.empty_like(key, dtype=torch.float32)
        grid_rms_k = (B * num_kv_heads * S,)
        rms_norm_kernel[grid_rms_k](
            key, k_norm_weight, key_norm, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0), sum_key.stride(0),
            rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4
        )

        # 7) Apply rotation to key_norm -> key_rot (bfloat16 output)
        key_rot = torch.empty_like(key_norm, dtype=torch.bfloat16)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 8) Update caches: write key_rot to key_cache[:, :, cache_position] and value to value_cache[:, :, cache_position]
        # Ensure key_rot and value are bfloat16 before storing
        key_rot_cast = key_rot.to(torch.bfloat16)
        value_cast = value.to(torch.bfloat16)
        grid_upd = (B * num_kv_heads * S,)
        update_cache_kernel[grid_upd](
            key_rot_cast, value_cast, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot_cast.stride(0), key_rot_cast.stride(1), key_rot_cast.stride(2),
            value_cast.stride(0), value_cast.stride(1), value_cast.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # Return query_rot and key_rot (same as original code), and updated caches
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
