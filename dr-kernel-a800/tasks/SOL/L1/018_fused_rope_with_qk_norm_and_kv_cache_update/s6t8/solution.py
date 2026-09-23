import torch
import triton
import triton.language as tl

@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, num_q_heads, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    acc = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    total_elems = H
    mean = acc / total_elems
    sum_ptr[pid] = mean  # store per-token mean of squares


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                     H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     eps: tl.constexpr,
                     BLOCK_H: tl.constexpr):
    # One program per (b, num_q_heads, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    mean = sum_ptr[pid]
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        w_offs = idx  # weight is 1D over head_dim
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + w_offs, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr,
                              cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr,
                              H: tl.constexpr,
                              H2: tl.constexpr,  # head_dim // 2
                              position_stride,  # 1 for [B,S]
                              cos_batch_stride, cos_s_stride, cos_h_stride,
                              sin_batch_stride, sin_s_stride, sin_h_stride,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    # Load position for this (b, s)
    pos = tl.load(position_ids_ptr + b * position_stride + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        half = H2

        # Build emb: first half = pos * inv_freq[idx], second half = duplicated
        emb_first = tl.load(inv_freq_ptr + idx, mask=idx < half, other=0.0).to(tl.float32)
        emb = tl.where(idx < half, emb_first, tl.load(inv_freq_ptr + (idx - half), mask=(idx >= half), other=0.0).to(tl.float32))
        emb = emb * pos

        # Compute sin/cos (Triton provides sin/cos)
        c = tl.cos(emb)
        s = tl.sin(emb)

        # Store cos/sin at [b, s, :]
        base = b * cos_batch_stride + s * cos_s_stride
        cos_out_offs = base + idx * cos_h_stride
        sin_out_offs = b * sin_batch_stride + s * sin_s_stride + idx * sin_h_stride

        tl.store(cos_ptr + cos_out_offs, c, mask=mask)
        tl.store(sin_ptr + sin_out_offs, s, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr,
                           out_ptr,
                           B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                           H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_out, h_stride_out, s_stride_out,
                           cos_batch_stride, cos_s_stride, cos_h_stride,
                           sin_batch_stride, sin_s_stride, sin_h_stride,
                           BLOCK_H: tl.constexpr):
    # One program per (b, num_q_heads, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        base_cos = b * cos_batch_stride + s * cos_s_stride
        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_h_stride, mask=mask, other=1.0).to(tl.float32)
        base_sin = b * sin_batch_stride + s * sin_s_stride
        sin_vals = tl.load(sin_ptr + base_sin + idx * sin_h_stride, mask=mask, other=1.0).to(tl.float32)

        first_half = idx < half
        x_first = tl.load(x_ptr + x_offs, mask=first_half, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + x_offs, mask=(~first_half), other=0.0).to(tl.float32)
        rotated = tl.where(first_half, x_first, -x_second)

        y = x * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        dest_ptr,
                        B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                        H: tl.constexpr,
                        batch_stride_kr, h_stride_kr, s_stride_kr, d_stride_kr,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        BLOCK_H: tl.constexpr):
    # Grid: (B*S, num_q_heads) — using query's num_q_heads for cache writes
    pid = tl.program_id(0)
    qh = tl.program_id(1)
    if pid >= B * S or qh >= num_q_heads:
        return
    b = pid // S
    s = pid % S

    dest = tl.load(dest_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        src_offs = b * batch_stride_kr + qh * h_stride_kr + s * s_stride_kr + idx * d_stride_kr
        src_vals = tl.load(key_rot_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(key_cache_ptr + b * batch_stride_kc + qh * h_stride_kc + dest * dest_stride + idx * d_stride_kc, src_vals, mask=mask)

        src_offs_val = b * batch_stride_v + qh * h_stride_v + s * s_stride_v + idx * d_stride_v
        src_vals_val = tl.load(value_ptr + src_offs_val, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + b * batch_stride_vc + qh * h_stride_vc + dest * dest_stride_vc + idx * d_stride_vc, src_vals_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value,
                position_ids,  # [B, S], int64
                key_cache, value_cache,
                cache_position,  # [S], int64
                q_norm_weight, k_norm_weight,  # [H], bfloat16
                inv_freq,  # [H//2], float32
                rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]  # typically 8

        # 1) Compute per-token sum of squares for RMSNorm (query)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](query, sum_sums_q,
                                   B, num_q_heads, S, H,
                                   query.stride(0), query.stride(1), query.stride(2),
                                   BLOCK_H=64)

        # 2) RMSNorm for query: normalize and apply q_norm_weight
        query_norm = torch.empty_like(query)
        grid_norm_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_q](query, q_norm_weight, query_norm, sum_sums_q,
                                     B, num_q_heads, S, H,
                                     query.stride(0), query.stride(1), query.stride(2),
                                     query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                     rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 3) Compute cos/sin for rotation per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        grid_rc = (B, S)
        H2 = H // 2
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, H2,
                                           1,  # position_stride
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=64, num_warps=4)

        # 4) Apply rotation to query_norm -> query_rot
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=64, num_warps=4)

        # 5) RMSNorm for key
        sum_sums_k = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_k](key, sum_sums_k,
                                   B, num_q_heads, S, H,
                                   key.stride(0), key.stride(1), key.stride(2),
                                   BLOCK_H=64)

        key_norm = torch.empty_like(key)
        grid_norm_k = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_k](key, k_norm_weight, key_norm, sum_sums_k,
                                     B, num_q_heads, S, H,
                                     key.stride(0), key.stride(1), key.stride(2),
                                     key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                     rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 6) Apply rotation to key_norm -> key_rot
        key_rot = torch.empty_like(key_norm)
        grid_rot_k = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rot,
                                          B, num_q_heads, S, H,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=64, num_warps=4)

        # 7) Update caches at positions cache_position[s] using query_rot (per token) and value (per token)
        grid_update = (B * S, num_q_heads)
        update_cache_kernel[grid_update](query_rot, value,
                                         key_cache, value_cache,
                                         cache_position,
                                         B, num_q_heads, S, H,
                                         query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), 1,
                                         value.stride(0), value.stride(1), value.stride(2), 1,
                                         key_cache.stride(0), key_cache.stride(1), 1, key_cache.stride(3),
                                         value_cache.stride(0), value_cache.stride(1), 1, value_cache.stride(3),
                                         BLOCK_H=64, num_warps=4)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
