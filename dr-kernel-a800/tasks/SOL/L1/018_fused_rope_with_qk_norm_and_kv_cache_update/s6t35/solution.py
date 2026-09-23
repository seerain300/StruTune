import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.int32, num_heads: tl.int32, S: tl.int32, H: tl.int32,
                    batch_stride_x, head_stride_x, s_stride_x, d_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    acc = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * d_stride_x
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_vals * x_vals, axis=0)
    # Store sum for (b, head, s)
    tl.store(sum_ptr + pid, acc)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.int32, num_heads: tl.int32, S: tl.int32, H: tl.int32,
                     batch_stride_x, head_stride_x, s_stride_x, d_stride_x,
                     batch_stride_out, head_stride_out, s_stride_out,
                     eps: tl.float32,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_ptr + pid)
    inv_rms = 1.0 / tl.sqrt(sum_val / H + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * d_stride_x
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x_vals * inv_rms) * w_vals
        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_x
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.int32, S: tl.int32, H: tl.int32, HALF: tl.int32,
                              pos_stride, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # Grid is (B, S): one program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    pos = tl.load(position_ids_ptr + b * pos_stride + s)

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # For first half, load inv_freq; second half duplicates first
        first_half = idx < HALF
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        emb = tl.where(first_half, emb_first, emb_first)  # duplicate to full H
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.int32, num_heads: tl.int32, S: tl.int32, H: tl.int32,
                           batch_stride_x, head_stride_x, s_stride_x, d_stride_x,
                           batch_stride_out, head_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x block
        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * d_stride_x
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated block correctly:
        # For i in [0, half): rotated[i] = -x[half + i]
        # For i in [half, H): rotated[i] = x[i - half]
        half = H // 2
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        rotated = tl.where(idx < half, -x_vals[half + tl.arange(0, BLOCK_H)], x_vals)
        rotated = tl.where(idx >= half, x_vals[tl.arange(0, BLOCK_H) - half], rotated)

        # Compute output: y = x * cos + rotated * sin
        y = x_vals * cos_vals + rotated * sin_vals

        # Store result
        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_x
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B: tl.int32, num_kv_heads: tl.int32, S: tl.int32, H: tl.int32,
                         batch_stride_k, head_stride_k, s_stride_k, d_stride_k,
                         batch_stride_v, head_stride_v, s_stride_v, d_stride_v,
                         batch_stride_ck, head_stride_ck, cache_stride_ck, d_stride_ck,
                         batch_stride_cv, head_stride_cv, cache_stride_cv, d_stride_vv,
                         BLOCK_H: tl.constexpr):
    # Grid is (B, num_kv_heads, S): one program per (b, kv_head, s)
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    s = tl.program_id(2)

    dest_pos = tl.load(cache_pos_ptr + s)

    # Load rotated key for (b, kv_head, s, :)
    k_offs = b * batch_stride_k + kv_head * head_stride_k + s * s_stride_k + tl.arange(0, BLOCK_H) * d_stride_k
    k_vals = tl.load(key_rot_ptr + k_offs, mask=tl.arange(0, BLOCK_H) < H, other=0.0).to(tl.float32)

    # Store into key_cache at (b, kv_head, dest_pos, :)
    ck_offs = b * batch_stride_ck + kv_head * head_stride_ck + dest_pos * cache_stride_ck + tl.arange(0, BLOCK_H) * d_stride_ck
    tl.store(key_cache_ptr + ck_offs, k_vals, mask=tl.arange(0, BLOCK_H) < H)

    # Load value for (b, kv_head, s, :)
    v_offs = b * batch_stride_v + kv_head * head_stride_v + s * s_stride_v + tl.arange(0, BLOCK_H) * d_stride_v
    v_vals = tl.load(value_ptr + v_offs, mask=tl.arange(0, BLOCK_H) < H, other=0.0).to(tl.float32)

    # Store into value_cache at (b, kv_head, dest_pos, :)
    cv_offs = b * batch_stride_cv + kv_head * head_stride_cv + dest_pos * cache_stride_cv + tl.arange(0, BLOCK_H) * d_stride_vv
    tl.store(value_cache_ptr + cv_offs, v_vals, mask=tl.arange(0, BLOCK_H) < H)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Make inputs contiguous for predictable strides
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query: query_norm = query / sqrt(mean(query^2) + eps) * q_norm_weight
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_q_sum = (B * num_q_heads * S,)
        # Strides for query
        d_stride_x = query.stride(3)
        rms_sum_kernel[grid_q_sum](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_x,
            BLOCK_H=64, num_warps=4
        )
        # Normalize
        grid_q_norm = (B * num_q_heads * S,)
        d_stride_out = query_norm.stride(3)
        rms_norm_kernel[grid_q_norm](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_x,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=64, num_warps=4
        )

        # 2) Compute rotation sin/cos per (b, s) -> cos[B, S, H], sin[B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        HALF = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](
            position_ids, inv_freq, cos, sin,
            B, S, H, HALF,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        d_stride_rot = query_rot.stride(3)
        apply_rotation_kernel[grid_rot](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), d_stride_x,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (bfloat16)
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty(B * num_kv_heads * S, dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_kv_heads * S,)
        d_stride_k = key.stride(3)
        rms_sum_kernel[grid_sum_k](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2), d_stride_k,
            BLOCK_H=64, num_warps=4
        )
        grid_norm_k = (B * num_kv_heads * S,)
        d_stride_vk = key_norm.stride(3)
        rms_norm_kernel[grid_norm_k](
            key, k_norm_weight, key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2), d_stride_k,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=64, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot
        key_rot = torch.empty_like(key_norm)
        grid_rot_k = (B * num_kv_heads * S,)
        d_stride_kout = key_rot.stride(3)
        apply_rotation_kernel[grid_rot_k](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), d_stride_k,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        grid_update = (B, num_kv_heads, S)
        d_stride_vk = key_rot.stride(3)
        d_stride_v = value.stride(3)
        cache_stride = key_cache.stride(2)  # along max_position_embeddings
        update_cache_kernel[grid_update](
            key_rot, value, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), d_stride_vk,
            value.stride(0), value.stride(1), value.stride(2), d_stride_v,
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), cache_stride,
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), cache_stride,
            BLOCK_H=64, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
