import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    sum_stride,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    total_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, total_sum)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_w, h_stride_w,
                     batch_stride_out, h_stride_out, s_stride_out,
                     eps: tl.constexpr,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_x = tl.load(sum_ptr + pid).to(tl.float32)
    mean = sum_x / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    inv_rms = inv_rms.to(tl.float32)

    # weight vector load
    w_offs = b * batch_stride_w + h * h_stride_w
    weight = tl.load(weight_ptr + w_offs + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H, other=0.0).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        y_vals = x_vals * inv_rms * weight
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y_vals, mask=mask)


@triton.jit
def rms_sum_key_kernel(k_ptr, sum_ptr,
                       B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                       batch_stride_k, h_stride_k, s_stride_k,
                       sum_stride,
                       BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    total_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        k_offs = b * batch_stride_k + h * h_stride_k + s * s_stride_k + idx
        k_vals = tl.load(k_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(k_vals * k_vals, axis=0)
    tl.store(sum_ptr + pid, total_sum)


@triton.jit
def rms_norm_key_kernel(k_ptr, weight_ptr, out_ptr, sum_ptr,
                        B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_k, h_stride_k, s_stride_k,
                        batch_stride_w, h_stride_w,
                        batch_stride_out, h_stride_out, s_stride_out,
                        eps: tl.constexpr,
                        BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_k = tl.load(sum_ptr + pid).to(tl.float32)
    mean = sum_k / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    inv_rms = inv_rms.to(tl.float32)

    # weight vector
    w_offs = b * batch_stride_w + h * h_stride_w
    weight = tl.load(weight_ptr + w_offs + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H, other=0.0).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        k_offs = b * batch_stride_k + h * h_stride_k + s * s_stride_k + idx
        k_vals = tl.load(k_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
        y_vals = k_vals * inv_rms * weight
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                              pos_stride,  # stride for position_ids [B, S]
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # Each program handles one (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)
    inv_len = H2  # H//2

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # i < H2: emb[i] = pos * inv_freq[i], else emb[i] = emb[i - H2]
        first_half = idx < H2
        second_half = idx >= H2
        # Load inv_freq only for first half
        inv_freq_vals = tl.load(inv_freq_ptr + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H2, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals  # [BLOCK_H]
        # Duplicate second half
        emb_second = emb_first
        emb = tl.where(first_half, emb_first, emb_second)

        c = tl.cos(emb)
        s_ = tl.sin(emb)
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_out, h_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base_cos = b * cos_stride0 + s * cos_stride1
    base_sin = b * sin_stride0 + s * sin_stride1

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin
        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_sin + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        # rotated = [-x_second, x_first], where second half is idx >= half, first half is idx < half
        x_first = x[:half]
        x_second = x[half:]
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        rotated = -x_second + x_first  # vectorized across BLOCK_H, but we need to index into x
        # To index correctly, compute masks:
        mask_first = idx < half
        mask_second = idx >= half
        x_first_vec = tl.load(x_ptr + x_offs, mask=mask_first, other=0.0).to(tl.float32)
        x_second_vec = tl.load(x_ptr + x_offs, mask=mask_second, other=0.0).to(tl.float32)
        rotated = tl.where(mask_first, x_first_vec, tl.zeros_like(x_first_vec)) + tl.where(mask_second, -x_second_vec, tl.zeros_like(x_second_vec))

        y = x * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        dest_ptr,
                        B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_kr, h_stride_kr, s_stride_kr, d_stride_kr,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        BLOCK_H: tl.constexpr):
    # Grid: (B * num_heads * S,)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // num_heads  # num_heads can be kv_heads or q_heads depending on context; here we use general h
    s = pid % S
    dest = tl.load(dest_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        src_offs = b * batch_stride_kr + h * h_stride_kr + s * s_stride_kr + idx * d_stride_kr
        src_vals = tl.load(key_rot_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(key_cache_ptr + b * batch_stride_kc + h * h_stride_kc + dest * dest_stride + idx * d_stride_kc, src_vals, mask=mask)

        src_offs_val = b * batch_stride_v + h * h_stride_v + s * s_stride_v + idx * d_stride_v
        src_vals_val = tl.load(value_ptr + src_offs_val, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + b * batch_stride_vc + h * h_stride_vc + dest * dest_stride_vc + idx * d_stride_vc, src_vals_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args order matches original: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        # Ensure inputs are contiguous (but we'll use strides anyway)
        # Allocate intermediates
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2

        # 1) RMSNorm for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](query, sum_sums_q,
                                   B, num_q_heads, S, H,
                                   query.stride(0), query.stride(1), query.stride(2),
                                   0,  # sum_stride unused, we pass as scalar
                                   BLOCK_H=64, num_warps=4)

        grid_norm_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_q](query, q_norm_weight, query_norm, sum_sums_q,
                                     B, num_q_heads, S, H,
                                     query.stride(0), query.stride(1), query.stride(2),
                                     q_norm_weight.stride(0), q_norm_weight.stride(1),
                                     query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                     rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 2) Compute cos/sin per (b, s)
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, H2,
                                           position_ids.stride(0),  # pos_stride
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=64, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=64, num_warps=4)

        # 4) RMSNorm for key
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_key_kernel[grid_sum_k](key, sum_sums_k,
                                       B, num_kv_heads, S, H,
                                       key.stride(0), key.stride(1), key.stride(2),
                                       0,  # sum_stride unused
                                       BLOCK_H=64, num_warps=4)

        grid_norm_k = (B * num_kv_heads * S,)
        rms_norm_key_kernel[grid_norm_k](key, k_norm_weight, key_norm, sum_sums_k,
                                         B, num_kv_heads, S, H,
                                         key.stride(0), key.stride(1), key.stride(2),
                                         k_norm_weight.stride(0), k_norm_weight.stride(1),
                                         key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                         rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 5) Apply rotation to key_norm -> key_rot
        key_rot = torch.empty_like(key_norm)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rot,
                                          B, num_kv_heads, S, H,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=64, num_warps=4)

        # 6) Update caches using cache_position rows
        # We only update for current inputs; cache_position has length S, mapping token s to cache row dest
        grid_upd = (B * num_kv_heads * S,)
        # Note: value has dtype matching input; we store original value (not rotated) into cache as per original code.
        update_cache_kernel[grid_upd](key_rot, value,
                                      key_cache, value_cache,
                                      cache_position,
                                      B, num_kv_heads, S, H,
                                      key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
                                      value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                                      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                                      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                                      BLOCK_H=64, num_warps=4)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
