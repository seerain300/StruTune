import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S
    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x * x, axis=0)
    tl.store(sum_ptr + b * sum_stride0, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     sum_stride0, eps: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S
    sum_val = tl.load(sum_ptr + b * sum_stride0).to(tl.float32)
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        in_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        out = (x * inv_rms) * w
        out = out.to(tl.bfloat16)
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, out, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr,
                              pos_stride, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    if (b >= B) or (s >= S):
        return
    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # Only first half needs inv_freq multiply
        first_mask = idx < HALF
        # Load inv_freq for first half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate to second half
        emb = tl.where(first_mask, emb_first, emb_first)  # emb[i] = emb_first[i] for i>=HALF as well
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_out, h_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x
        in_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated part: rotated[i] = -x[HALF + i] if i < HALF, else x[i - HALF]
        rotated = tl.zeros_like(x)
        for i in range(0, BLOCK_H):
            if i < HALF:
                rotated[i] = -tl.load(x_ptr + in_offs + (i + HALF), mask=True, other=0.0).to(tl.float32)
            else:
                rotated[i] = tl.load(x_ptr + in_offs + (i - HALF), mask=True, other=0.0).to(tl.float32)

        out_vals = x * cos_vals + rotated * sin_vals
        out_vals = out_vals.to(tl.bfloat16)
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, out_vals, mask=mask)


@triton.jit
def update_cache_kernel(x_ptr, cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_x, h_stride_x, s_stride_x,
                         batch_stride_cache, h_stride_cache, pos_stride_cache,
                         BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S
    dest_pos = tl.load(cache_pos_ptr + s).to(tl.int32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.bfloat16)
        dest_offs = b * batch_stride_cache + h * h_stride_cache + dest_pos * pos_stride_cache + idx
        tl.store(cache_ptr + dest_offs, x_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        B_k, num_kv_heads, S_k, H_k = key.shape
        assert B == B_k and S == S_k and H == H_k, "Incompatible shapes"
        HALF = H // 2
        device = query.device

        # 1) RMSNorm for query -> query_norm (bf16)
        query_norm = torch.empty_like(query, dtype=torch.float16)  # we'll compute in fp32 and cast
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](query, sum_sums_q,
                                   B, num_q_heads, S, H,
                                   query.stride(0), query.stride(1), query.stride(2),
                                   sum_sums_q.stride(0),
                                   BLOCK_H=128, num_warps=4)

        rms_norm_kernel[grid_sum_q](query, q_norm_weight, query_norm, sum_sums_q,
                                    B, num_q_heads, S, H,
                                    query.stride(0), query.stride(1), query.stride(2),
                                    query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                    sum_sums_q.stride(0), eps=rms_norm_eps, BLOCK_H=128, num_warps=4)

        # 2) Compute sin/cos rotation [B, S, H] (float32)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, HALF,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=128, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot (bf16)
        query_rot = torch.empty_like(query_norm, dtype=torch.float16)
        grid_rot_q = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot_q](query_norm, cos, sin, query_rot,
                                          B, num_q_heads, S, H, HALF,
                                          query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                          query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=128, num_warps=4)

        # 4) RMSNorm for key -> key_norm (bf16)
        key_norm = torch.empty_like(key, dtype=torch.float16)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](key, sum_sums_k,
                                   B, num_kv_heads, S, H,
                                   key.stride(0), key.stride(1), key.stride(2),
                                   sum_sums_k.stride(0),
                                   BLOCK_H=128, num_warps=4)

        rms_norm_kernel[grid_sum_k](key, k_norm_weight, key_norm, sum_sums_k,
                                    B, num_kv_heads, S, H,
                                    key.stride(0), key.stride(1), key.stride(2),
                                    key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                    sum_sums_k.stride(0), eps=rms_norm_eps, BLOCK_H=128, num_warps=4)

        # 5) Apply rotation to key_norm -> key_rot (bf16)
        key_rot = torch.empty_like(key_norm, dtype=torch.float16)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rot,
                                          B, num_kv_heads, S, H, HALF,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=128, num_warps=4)

        # 6) Update key_cache and value_cache at cache_position (bf16, no inplace mutation of inputs)
        # key_cache: copy key_rot into rows indicated by cache_position
        # value_cache: copy value into rows indicated by cache_position
        # Allocate temp to hold per (b, kv_head, s) value (same shape as value)
        grid_upd = (B * num_kv_heads * S,)
        update_cache_kernel[grid_upd](key_rot, key_cache, cache_position,
                                      B, num_kv_heads, S, H,
                                      key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                                      BLOCK_H=128, num_warps=4)

        grid_upd_v = (B * num_kv_heads * S,)
        update_cache_kernel[grid_upd_v](value, value_cache, cache_position,
                                        B, num_kv_heads, S, H,
                                        value.stride(0), value.stride(1), value.stride(2),
                                        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                                        BLOCK_H=128, num_warps=4)

        # Return as original: (query_rotated, key_rotated, key_cache, value_cache)
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
