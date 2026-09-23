import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     rms_norm_eps,
                     BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_ptr + pid)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + rms_norm_eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        offs_w = idx  # weight is 1D contiguous
        w_vals = tl.load(weight_ptr + offs_w, mask=mask, other=1.0).to(tl.float32)

        y_vals = x_vals * inv_rms * w_vals
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                              pos_stride,
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

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half = idx < H2
        # For first half, emb = pos * inv_freq[idx]
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_half, other=0.0).to(tl.float32)
        emb = tl.where(first_half, pos * inv_freq_vals, pos * inv_freq_vals)  # duplicate second half

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

        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        rotated_first_mask = idx < half
        rotated_second_mask = idx >= half
        # First half: rotated[i] = -x[i + half]
        rotated_first = -tl.load(x_ptr + offs_x, mask=rotated_first_mask, other=0.0).to(tl.float32)
        # Second half: rotated[i] = x[i - half]
        rotated_second = tl.load(x_ptr + b * batch_stride_x + h * h_stride_x + s * s_stride_x + (idx - half), mask=rotated_second_mask, other=0.0).to(tl.float32)
        rotated = tl.where(rotated_first_mask, rotated_first, rotated_second)

        y = x_vals * cos_vals + rotated * sin_vals
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def update_cache_kernel(rot_key_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_rk, h_stride_rk, s_stride_rk, d_stride_rk,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, d_stride_kc,
                        batch_stride_vc, h_stride_vc, d_stride_vc,
                        cache_position_ptr,
                        BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kvh = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest = tl.load(cache_position_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        src_offs = b * batch_stride_rk + kvh * h_stride_rk + s * s_stride_rk + idx * d_stride_rk
        src_vals = tl.load(rot_key_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)

        dst_offs = b * batch_stride_kc + kvh * h_stride_kc + dest * d_stride_kc + idx * d_stride_kc
        tl.store(key_cache_ptr + dst_offs, src_vals, mask=mask)

        src_offs_val = b * batch_stride_v + kvh * h_stride_v + s * s_stride_v + idx * d_stride_v
        src_vals_val = tl.load(value_ptr + src_offs_val, mask=mask, other=0.0).to(tl.float32)

        dst_offs_val = b * batch_stride_vc + kvh * h_stride_vc + dest * d_stride_vc + idx * d_stride_vc
        tl.store(value_cache_ptr + dst_offs_val, src_vals_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        _, num_kv_heads, _, _ = key.shape
        # We'll run RMSNorm on query and key (assuming key is [B, num_kv_heads, S, H])
        # 1) RMSNorm for query -> query_norm
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid_q = (B * num_q_heads * S,)
        BLOCK_H = 64  # works for H up to 128 in this workload
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_q](query, q_norm_weight, query_norm, sum_sums_q,
                                B, num_q_heads, S, H,
                                query.stride(0), query.stride(1), query.stride(2),
                                query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 2) Compute cos/sin per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, H2,
                                           position_ids.stride(0),  # pos_stride for [B, S]
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=BLOCK_H, num_warps=4)

        # 4) RMSNorm for key -> key_norm
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](key, sum_sums_k,
                                   B, num_kv_heads, S, H,
                                   key.stride(0), key.stride(1), key.stride(2),
                                   BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_sum_k](key, k_norm_weight, key_norm, sum_sums_k,
                                    B, num_kv_heads, S, H,
                                    key.stride(0), key.stride(1), key.stride(2),
                                    key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                    rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 5) Rotate key_norm -> key_rot
        key_rot = torch.empty_like(key_norm)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rot,
                                          B, num_kv_heads, S, H,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 6) Update caches using cache_position (S vector) -> this overwrites existing caches
        # Note: cache_position must be on device and int32 for Triton
        # Here we update only key/value caches at positions [cache_len : cache_len + S]
        # We cannot know cache_len from inputs, but the original run uses cache_position, so we use it.
        # Ensure cache_position is int32 and device-matched
        cache_pos_int = cache_position.to(dtype=torch.int32, device=query.device)

        update_cache_kernel[grid_rot_k](key_rot, value, key_cache, value_cache,
                                        B, num_kv_heads, S, H,
                                        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
                                        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                                        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                                        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                                        cache_pos_int,
                                        BLOCK_H=BLOCK_H, num_warps=4)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
