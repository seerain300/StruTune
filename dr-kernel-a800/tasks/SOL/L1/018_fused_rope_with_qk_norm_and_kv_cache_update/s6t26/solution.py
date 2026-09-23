import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    eps: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    batch_stride_out, h_stride_out, s_stride_out,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_sums_ptr + pid)
    Hf = tl.full((), H, tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_sq / Hf + eps)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * w_vals * inv_rms
        out_offs = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y_vals.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              pos_stride,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * pos_stride).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        i2 = idx // 2
        half = H // 2
        first_mask = idx < half
        emb_first = pos * tl.load(inv_freq_ptr + i2, mask=first_mask, other=0.0).to(tl.float32)
        emb = tl.where(first_mask, emb_first, emb_first)  # second half repeats first half
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
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
    # One program per (b, head, s)
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
        x_offs = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # Rotate half: rotated[i] = -x[half + i] for i < half, and rotated[half + i] = x[i] for i < half
        for i in range(0, half):
            rotated[i] = -x_vals[half + i]
            rotated[half + i] = x_vals[i]
        y_vals = x_vals * cos_vals + rotated * sin_vals
        out_offs = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y_vals.to(tl.bfloat16), mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_position_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_q, h_stride_q, s_stride_q,
                         batch_stride_kc, h_stride_kc, s_stride_kc,
                         batch_stride_vc, h_stride_vc, s_stride_vc,
                         cp_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_head = (pid % (num_kv_heads * S)) // S
    s = pid % S
    cp = tl.load(cache_position_ptr + s * cp_stride).to(tl.int32)  # cache row index

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # Read rotated query for this (b, kv_head, s)
        q_offs = b * batch_stride_q + kv_head * h_stride_q + s * s_stride_q + idx
        q_vals = tl.load(query_rot_ptr + q_offs, mask=mask, other=0.0).to(tl.bfloat16)

        # Write to key_cache at [b, kv_head, cp, :]
        kc_offs = b * batch_stride_kc + kv_head * h_stride_kc + cp * s_stride_kc + idx
        tl.store(key_cache_ptr + kc_offs, q_vals, mask=mask)

        # Write value[b, kv_head, s, :] to value_cache at [b, kv_head, cp, :]
        v_offs = b * batch_stride_vc + kv_head * h_stride_vc + s * s_stride_vc + idx
        v_vals = tl.load(value_ptr + v_offs, mask=mask, other=0.0).to(tl.bfloat16)
        vc_offs = b * batch_stride_vc + kv_head * h_stride_vc + cp * s_stride_vc + idx
        tl.store(value_cache_ptr + vc_offs, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMS sum for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 2) RMSNorm for query -> query_norm (bfloat16)
        query_norm = torch.empty_like(query)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            float(rms_norm_eps),
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 3) Compute rotation cos/sin per (b, s) in float32 -> [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 4) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # Optional: same for key if needed (not in original returns, but kept for robustness)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=64,
            num_warps=4
        )
        key_rot = torch.empty_like(key)
        apply_rotation_kernel[(B * num_kv_heads * S,)](
            key, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 5) Update caches using rotated query and original value at cache_position
        update_cache_kernel[(B * num_kv_heads * S,)](
            query_rot, value, key_cache, value_cache,
            cache_position,
            B, num_kv_heads, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position.stride(0),
            BLOCK_H=64,
            num_warps=4
        )

        # Return as original function: rotated query, rotated key, updated caches
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
