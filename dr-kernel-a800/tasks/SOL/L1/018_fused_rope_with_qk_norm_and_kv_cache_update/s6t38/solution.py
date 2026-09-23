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
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals
        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_x
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr,
                              cos_ptr, sin_ptr,
                              B: tl.int32, S: tl.int32, H: tl.int32,
                              pos_stride, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)

    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        first_half = idx < half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate second half from first half
        emb = tl.where(first_half, emb_first, emb_first)
        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


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

        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * head_stride_x
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        # Build rotated block:
        # For i in [0, half): rotated[i] = -x[half + i]
        # For i in [half, H): rotated[i] = x[i - half]
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in range(BLOCK_H):
            if (off + i) < half:
                rotated[i] = -x_vals[half + (off + i)]
            else:
                rotated[i] = x_vals[(off + i) - half]

        y = x_vals * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * head_stride_out
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(key_out_ptr, value_ptr,
                         key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B: tl.int32, num_kv_heads: tl.int32, S: tl.int32, H: tl.int32,
                         kbatch_stride, khead_stride, ks_stride, kd_stride,
                         vbatch_stride, vhead_stride, vs_stride, vd_stride,
                         cpos_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    b = pid // (num_kv_heads * S)
    kv_head = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest = tl.load(cache_pos_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * kbatch_stride + kv_head * khead_stride + s * ks_stride + idx * kd_stride
        vals = tl.load(key_out_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        kc_offs = b * key_cache_ptr.stride(0) + kv_head * key_cache_ptr.stride(1) + dest * key_cache_ptr.stride(2) + idx * key_cache_ptr.stride(3)
        tl.store(key_cache_ptr + kc_offs, vals, mask=mask)

        v_offs = b * vbatch_stride + kv_head * vhead_stride + s * vs_stride + idx * vd_stride
        v_vals = tl.load(value_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
        vc_offs = b * value_cache_ptr.stride(0) + kv_head * value_cache_ptr.stride(1) + dest * value_cache_ptr.stride(2) + idx * value_cache_ptr.stride(3)
        tl.store(value_cache_ptr + vc_offs, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        # Ensure contiguity for predictable strides
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

        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        H = query.shape[3]

        # 1) RMSNorm for query -> query_norm
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        d_stride_q = query.stride(3)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_q,
            BLOCK_H=128, num_warps=4
        )
        d_stride_qn = query_norm.stride(3)
        grid_norm_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_q](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2), d_stride_q,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=128, num_warps=4
        )

        # 2) Compute rotation sin/cos: cos/sin of shape [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        grid_rc = (B * S,)
        rotate_sin_cos_kernel_b_s[grid_rc](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0), cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        d_stride_qrot = query_rot.stride(3)
        grid_rot_q = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot_q](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (bfloat16)
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty(B * key.shape[1] * S, dtype=torch.float32, device=key.device)
        num_kv_heads = key.shape[1]
        d_stride_k = key.stride(3)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2), d_stride_k,
            BLOCK_H=128, num_warps=4
        )
        d_stride_kn = key_norm.stride(3)
        grid_norm_k = (B * num_kv_heads * S,)
        rms_norm_kernel[grid_norm_k](
            key, k_norm_weight, key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2), d_stride_k,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=128, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot
        key_rot = torch.empty_like(key_norm)
        d_stride_krot = key_rot.stride(3)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        grid_update = (B, num_kv_heads, S)
        update_cache_kernel[grid_update](
            key_rot, value, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            cache_position.stride(0),
            BLOCK_H=128, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
