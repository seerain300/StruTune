import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, seq_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_s = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_x + head * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_s += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, sum_s)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, y_ptr, inv_rms_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, head_stride_x, seq_stride_x,
                     batch_stride_y, head_stride_y, seq_stride_y,
                     BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    inv_rms = tl.load(inv_rms_ptr + pid)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        offs_y = b * batch_stride_y + head * head_stride_y + s * seq_stride_y + idx
        tl.store(y_ptr + offs_y, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              batch_stride_pos, seq_stride_pos,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * batch_stride_pos + s * seq_stride_pos).to(tl.float32)
    # Build emb vector of length H
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        half = H // 2
        # For first half: emb[i] = pos * inv_freq[i//2]
        first_mask = idx < half
        inv_idx = idx // 2  # works for H even; idx < half ensures idx//2 < half
        inv_vals = tl.load(inv_freq_ptr + inv_idx, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_vals
        # For second half: duplicate first half
        emb = tl.where(first_mask, emb_first, emb_first)

        # Compute cos and sin
        c = tl.cos(emb)
        s_ = tl.sin(emb)

        # Store to [b, s, idx]
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, head_stride_x, seq_stride_x,
                           batch_stride_out, head_stride_out, seq_stride_out,
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
    inv_rms_x = 1.0  # not used here; x_ptr already normalized by caller
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # Load x
        offs_x = b * batch_stride_x + head * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # Build rotated according to idx range
        for i in range(0, BLOCK_H):
            if (idx[i] < half):
                rotated[i] = -x_vals[half + i]
            else:
                rotated[i] = x_vals[idx[i] - half]
        y_vals = x_vals * cos_vals + rotated * sin_vals

        # Store
        offs_out = b * batch_stride_out + head * head_stride_out + s * seq_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(src_ptr, dst_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         src_batch_stride, src_head_stride, src_seq_stride,
                         dst_batch_stride, dst_head_stride, dst_seq_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    dest_row = tl.load(cache_pos_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load from src (b, head, s, :)
        offs_src = b * src_batch_stride + head * src_head_stride + s * src_seq_stride + idx
        vals = tl.load(src_ptr + offs_src, mask=mask, other=0.0).to(tl.float32)

        # Store to dst (b, head, dest_row, :)
        offs_dst = b * dst_batch_stride + head * dst_head_stride + dest_row * dst_seq_stride + idx
        tl.store(dst_ptr + offs_dst, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query: query_norm
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=128, num_warps=4
        )

        query_norm = torch.empty_like(query)
        inv_rms_q = torch.empty_like(sum_sums_q, dtype=torch.float32, device=query.device)
        # Compute inv_rms_q on host from sum_sums_q
        inv_rms_q = 1.0 / torch.sqrt((sum_sums_q / H) + rms_norm_eps)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm,
            inv_rms_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 2) Compute cos/sin per (b, s) for rotation and store [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B * S,)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0), position_ids.stride(1),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 4) RMSNorm for key: key_norm
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=128, num_warps=4
        )

        key_norm = torch.empty_like(key)
        inv_rms_k = torch.empty_like(sum_sums_k, dtype=torch.float32, device=key.device)
        inv_rms_k = 1.0 / torch.sqrt((sum_sums_k / H) + rms_norm_eps)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm,
            inv_rms_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot (bfloat16)
        key_rot = torch.empty_like(key_norm)
        apply_rotation_kernel[(B * num_kv_heads * S,)](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        # For key_cache update
        update_cache_kernel[(B * num_kv_heads * S,)](
            key_rot, key_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # For value_cache
        update_cache_kernel[(B * num_kv_heads * S,)](
            value, value_cache, cache_position,
            B, num_kv_heads, S, H,
            value.stride(0), value.stride(1), value.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            BLOCK_H=128, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
