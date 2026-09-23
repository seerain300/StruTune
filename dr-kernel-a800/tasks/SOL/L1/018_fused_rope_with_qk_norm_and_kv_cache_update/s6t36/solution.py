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
        weight = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y_vals = x_vals * inv_rms * weight
        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_x
        tl.store(out_ptr + out_offs, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.int32, S: tl.int32, H: tl.int32,
                              pos_stride, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * pos_stride + s)
    base = b * cos_stride0 + s * cos_stride1

    half = H // 2
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        c = tl.cos(emb_first)
        s_ = tl.sin(emb_first)
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)
        # Duplicate to second half
        full_idx = idx + half
        tl.store(cos_ptr + base + full_idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + full_idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.int32, num_heads: tl.int32, S: tl.int32, H: tl.int32,
                           batch_stride_x, head_stride_x, s_stride_x,
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
        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * d_stride_x  # use d_stride_x for vector indexing along H
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # rotated = -x[half + idx] for idx < half; rotated = x[idx - half] for idx >= half
        half = H // 2
        rotated = tl.zeros_like(x_vals, dtype=tl.float32)
        for i in range(0, BLOCK_H):
            ri = off + i
            if ri < half:
                rotated[ri] = -x_vals[ri + half]
            else:
                rotated[ri] = x_vals[ri - half]

        y = x_vals * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_x
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(rotated_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.int32, num_heads: tl.int32, S: tl.int32, H: tl.int32,
                         batch_stride_in, head_stride_in, s_stride_in,
                         batch_stride_kc, head_stride_kc, pos_stride_kc,
                         batch_stride_vc, head_stride_vc, pos_stride_vc,
                         BLOCK_H: tl.constexpr):
    # One program per (b, head, s), write to key_cache[b, head, cache_pos[s], :]
    pid = tl.program_id(0)
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    cp = tl.load(cache_pos_ptr + s)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        in_offs = b * batch_stride_in + head * head_stride_in + s * s_stride_in + idx * head_stride_in
        vals = tl.load(rotated_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)

        kc_offs = b * batch_stride_kc + head * head_stride_kc + cp * pos_stride_kc + idx * head_stride_kc
        tl.store(key_cache_ptr + kc_offs, vals, mask=mask)

        vc_offs = b * batch_stride_vc + head * head_stride_vc + cp * pos_stride_vc + idx * head_stride_vc
        vals_v = tl.load(value_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + vc_offs, vals_v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure contiguity for predictable strides
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()
        cache_position = cache_position.contiguous()

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        assert value.shape == (B, num_kv_heads, S, H), "value shape must match (B, num_kv_heads, S, H)"
        assert position_ids.shape == (B, S), "position_ids shape must be (B, S)"
        assert cache_position.shape == (S,), "cache_position shape must be (S,)"

        # 1) RMSNorm for query -> query_norm
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        d_stride_q = query.stride(3)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_q,
            BLOCK_H=64, num_warps=4
        )
        d_stride_qout = query_norm.stride(3)
        grid_norm_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_q](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_q,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=64, num_warps=4
        )

        # 2) Compute rotation angles cos/sin per (b, s) using only first half of inv_freq, duplicate to full H
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        half = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        d_stride_x = query_norm.stride(3)
        d_stride_out = query_rot.stride(3)
        apply_rotation_kernel[grid_rot](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (bfloat16)
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty(B * num_kv_heads * S, dtype=torch.float32, device=key.device)
        d_stride_k = key.stride(3)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2), d_stride_k,
            BLOCK_H=64, num_warps=4
        )
        d_stride_vk = key_norm.stride(3)
        grid_norm_k = (B * num_kv_heads * S,)
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
        d_stride_kin = key_norm.stride(3)
        d_stride_kout = key_rot.stride(3)
        apply_rotation_kernel[grid_rot_k](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 6) Update caches using cache_position
        grid_update = (B, num_kv_heads, S)
        d_stride_in = key_rot.stride(3)  # rotated input has same D layout
        kc_bs, kc_hs, kc_ps = key_cache.stride(0), key_cache.stride(1), key_cache.stride(2)
        vc_bs, vc_hs, vc_ps = value_cache.stride(0), value_cache.stride(1), value_cache.stride(2)
        update_cache_kernel[grid_update](
            key_rot, value, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            kc_bs, kc_hs, kc_ps,
            value_cache.stride(0), value_cache.stride(1), vc_ps,
            BLOCK_H=64, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
