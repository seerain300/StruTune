import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, s_stride_x,
                    sum_stride0, sum_stride1, sum_stride2,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    total = tl.zeros((), dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x_vals * x_vals, axis=0)
    out_off = b * sum_stride0 + h * sum_stride1 + s * sum_stride2
    tl.store(sum_ptr + out_off, total)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, s_stride_x,
                    batch_stride_out, head_stride_out, s_stride_out,
                    weight_stride,
                    sum_stride0, sum_stride1, sum_stride2,
                    eps: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_ptr + (b * sum_stride0 + h * sum_stride1 + s * sum_stride2))
    inv_rms = 1.0 / tl.sqrt(sum_val / H + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        w_offs = idx
        w_vals = tl.load(weight_ptr + w_offs * weight_stride, mask=mask, other=0.0).to(tl.float32)

        out_offs = b * batch_stride_out + h * head_stride_out + s * s_stride_out + idx
        y_vals = x_vals * inv_rms * w_vals
        tl.store(out_ptr + out_offs, y_vals.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              pos_stride0, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * pos_stride0 + s * pos_stride0).to(tl.float32)  # position_ids[b, s]

    # Compute emb[i] = pos * inv_freq[i//2] for i < H//2 and duplicate for i >= H//2
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_mask = idx < half
        # Load inv_freq only for first half
        inv_off = idx // 2
        inv_vals = tl.load(inv_freq_ptr + inv_off, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_vals
        # Duplicate for second half
        emb = tl.where(first_mask, emb_first, emb_first)
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, head_stride_x, s_stride_x,
                           batch_stride_out, head_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base_out = b * batch_stride_out + h * head_stride_out + s * s_stride_out
    base_cos = b * cos_stride0 + s * cos_stride1

    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_cos + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # rotated = [-x[half:], x[:half]]
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # Fill rotated with correct mapping
        for j in range(BLOCK_H):
            i = off + j
            if i < half:
                rotated[j] = -tl.load(x_ptr + x_offs - half, mask=mask, other=0.0).to(tl.float32)[j]
            else:
                rotated[j] = tl.load(x_ptr + x_offs - half, mask=mask, other=0.0).to(tl.float32)[j - half]

        y_vals = x_vals * cos_vals + rotated * sin_vals
        tl.store(out_ptr + base_out + idx, y_vals.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def update_cache_kernel(key_cache_ptr, value_cache_ptr, rotated_ptr, value_ptr,
                         cache_position_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         b_stride_cache, h_stride_cache, pos_stride_cache, h_stride_cache2,
                         b_stride_val, h_stride_val, s_stride_val,
                         batch_stride_rotated, head_stride_rotated, s_stride_rotated,
                         BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    # Destination cache position is scalar per s
    dest_pos = tl.load(cache_position_ptr + s).to(tl.int32)

    # Copy rotated key into key_cache[b, h, dest_pos, :]
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        rotated_offs = b * batch_stride_rotated + h * head_stride_rotated + s * s_stride_rotated + idx
        rotated_vals = tl.load(rotated_ptr + rotated_offs, mask=mask, other=0.0).to(tl.float32)

        cache_offs = b * b_stride_cache + h * h_stride_cache + dest_pos * pos_stride_cache + idx * h_stride_cache2
        tl.store(key_cache_ptr + cache_offs, rotated_vals.to(key_cache_ptr.dtype.element_ty), mask=mask)

    # Copy value into value_cache[b, h, dest_pos, :]
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        val_offs = b * b_stride_val + h * h_stride_val + s * s_stride_val + idx
        val_vals = tl.load(value_ptr + val_offs, mask=mask, other=0.0).to(tl.float32)

        cache_offs = b * b_stride_cache + h * h_stride_cache + dest_pos * pos_stride_cache + idx * h_stride_cache2
        tl.store(value_cache_ptr + cache_offs, val_vals.to(value_cache_ptr.dtype.element_ty), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure device is CUDA
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda and \
               key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda and \
               q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query -> query_norm (bfloat16 output)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            BLOCK_H=128, num_warps=4
        )
        query_norm = torch.empty_like(query, dtype=torch.float32)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight.to(torch.float32), query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            k_norm_weight.stride(0),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=128, num_warps=4
        )

        # 2) Compute rotation embedding cos/sin per (b, s) -> [B, S, H] float32
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0), cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty_like(query_norm, dtype=torch.bfloat16)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (float32 output)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            BLOCK_H=128, num_warps=4
        )
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=key.device)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight.to(torch.float32), key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=128, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot (bfloat16 output)
        key_rot = torch.empty((B, num_kv_heads, S, H), dtype=torch.bfloat16, device=key.device)
        apply_rotation_kernel[(B * num_kv_heads * S,)](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot, value_cache[:, :, cache_position] = value
        # Note: value is [B, num_kv_heads, S, H], we only write the s-th slice to dest_pos = cache_position[s]
        update_cache_kernel[(B * num_kv_heads * S,)](
            key_cache, value_cache, key_rot, value,
            cache_position,
            B, num_kv_heads, S, H,
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            BLOCK_H=128, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
