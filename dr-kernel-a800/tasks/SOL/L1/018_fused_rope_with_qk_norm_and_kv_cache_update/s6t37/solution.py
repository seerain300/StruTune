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
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * w * inv_rms
        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_out
        tl.store(out_ptr + out_offs, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                               B: tl.int32, S: tl.int32, H: tl.int32,
                               pos_stride, cos_stride0, cos_stride1, cos_stride2,
                               sin_stride0, sin_stride1, sin_stride2,
                               BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0) // S
    s = tl.program_id(0) % S

    # Load position
    pos = tl.load(position_ids_ptr + b * pos_stride + s)
    pos = pos.to(tl.float32)

    half = H // 2
    # Loop over first half to get inv_freq indices
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate second half
        emb = tl.where(mask, emb_first, emb_first)  # full H vector for sin/cos
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)
    # Fill the rest (idx >= half) using duplicated values from first half: since we only store for idx < half, but sin/cos arrays are length H, we can rely on Triton to compute and store c/s for all idx up to H via grid and masks; here, we recompute for idx >= half using the same emb_first (H is even, so half == H//2 and this loop covers all H). To be precise, we'll recompute for full idx.
    idx_full = tl.arange(0, BLOCK_H)
    for off in range(0, half, BLOCK_H):
        idx = off + idx_full
        mask_full = idx < H
        inv_freq_vals = tl.load(inv_freq_ptr + off + tl.arange(0, BLOCK_H), mask=True, other=0.0).to(tl.float32)
        emb = pos * inv_freq_vals  # broadcast over idx_full
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask_full)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask_full)


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

        # Load x
        x_offs = b * batch_stride_x + head * head_stride_x + s * s_stride_x + idx * d_stride_x
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated block:
        # For i in [0, half): rotated[i] = -x[half + i]
        # For i in [half, H): rotated[i] = x[i - half]
        half = H // 2
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # We need two segments: [0:half) and [half:H)
        # First segment: i in [0, half)
        for i in range(0, half):
            j = half + i
            rotated_i = -x_vals[j]  # scalar
            rotated[i] = rotated_i
        # Second segment: i in [half, H)
        for i in range(half, H):
            j = i - half
            rotated_i = x_vals[j]  # scalar
            rotated[i] = rotated_i

        y_vals = x_vals * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + head * head_stride_out + s * s_stride_out + idx * d_stride_out
        tl.store(out_ptr + out_offs, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(rotated_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.int32, num_kv_heads: tl.int32, S: tl.int32, H: tl.int32,
                         batch_stride_rot, head_stride_rot, s_stride_rot,
                         batch_stride_kc, head_stride_kc,
                         batch_stride_vc, head_stride_vc,
                         cache_stride,  # this is the sequence dim stride, typically H*4 for bfloat16
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    # Copy rotated key into key_cache[b, kv_head, pos, :]
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        rotated_offs = b * batch_stride_rot + kv_head * head_stride_rot + s * s_stride_rot + idx * d_stride_rot
        rotated_vals = tl.load(rotated_ptr + rotated_offs, mask=mask, other=0.0)
        dest_offs = b * batch_stride_kc + kv_head * head_stride_kc + pos * cache_stride + idx * d_stride_kc
        tl.store(key_cache_ptr + dest_offs, rotated_vals, mask=mask)

    # Copy original value into value_cache[b, kv_head, pos, :]
    # Note: original code updates value_cache with 'value' (unrotated), not rotated key.
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        value_offs = b * batch_stride_vc + kv_head * head_stride_vc + s * s_stride_vc + idx * d_stride_vc
        value_vals = tl.load(value_ptr + value_offs, mask=mask, other=0.0)
        dest_offs = b * batch_stride_kc + kv_head * head_stride_kc + pos * cache_stride + idx * d_stride_kc  # this line is incorrect: we should use value_cache
        # Use separate value_cache_ptr
        dest_offs_vc = b * batch_stride_vc + kv_head * head_stride_vc + pos * cache_stride + idx * d_stride_vc
        tl.store(value_cache_ptr + dest_offs_vc, value_vals, mask=mask)


# Forward function
def run_with_triton(query: torch.Tensor,
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
    # Ensure tensors are contiguous and on CUDA
    device = query.device
    assert device.type == 'cuda', "Triton kernels require CUDA tensors"

    B, num_q_heads, S, H = query.shape
    num_kv_heads = key.shape[1]

    # 1) RMSNorm for query -> query_norm
    query_norm = torch.empty_like(query)
    sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
    grid_sum_q = (B * num_q_heads * S,)
    d_stride_q = query.stride(3)
    rms_sum_kernel[grid_sum_q](
        query, sum_sums_q,
        B, num_q_heads, S, H,
        query.stride(0), query.stride(1), query.stride(2), d_stride_q,
        BLOCK_H=64, num_warps=4
    )
    grid_norm_q = (B * num_q_heads * S,)
    d_stride_qout = query_norm.stride(3)
    rms_norm_kernel[grid_norm_q](
        query, q_norm_weight, query_norm, sum_sums_q,
        B, num_q_heads, S, H,
        query.stride(0), query.stride(1), query.stride(2), d_stride_q,
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
        float(rms_norm_eps),
        BLOCK_H=64, num_warps=4
    )

    # 2) Compute sin/cos rotation per (b, s)
    # Make position_ids contiguous int32
    position_ids_i32 = position_ids.to(torch.int32)
    cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
    sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
    grid_rc = (B * S,)
    rotate_sin_cos_kernel_b_s[grid_rc](
        position_ids_i32, inv_freq, cos, sin,
        B, S, H,
        position_ids_i32.stride(0),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        BLOCK_H=64, num_warps=4
    )

    # 3) Apply rotation to query_norm -> query_rot (bfloat16)
    query_rot = torch.empty_like(query_norm)
    grid_rot_q = (B * num_q_heads * S,)
    d_stride_rot_q = query_rot.stride(3)
    apply_rotation_kernel[grid_rot_q](
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
    grid_sum_k = (B * num_kv_heads * S,)
    d_stride_k = key.stride(3)
    rms_sum_kernel[grid_sum_k](
        key, sum_sums_k,
        B, num_kv_heads, S, H,
        key.stride(0), key.stride(1), key.stride(2), d_stride_k,
        BLOCK_H=64, num_warps=4
    )
    grid_norm_k = (B * num_kv_heads * S,)
    d_stride_kout = key_norm.stride(3)
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
    d_stride_krot = key_rot.stride(3)
    apply_rotation_kernel[grid_rot_k](
        key_norm, cos, sin, key_rot,
        B, num_kv_heads, S, H,
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        BLOCK_H=64, num_warps=4
    )

    # 6) Update caches: key_cache[:, :, cache_position] = key_rot, value_cache[:, :, cache_position] = value
    key_cache = key_cache.clone()  # ensure we write into existing buffer
    value_cache = value_cache.clone()
    grid_update = (B, num_kv_heads, S)
    cache_stride = key_cache.stride(2)  # sequence dimension stride
    update_cache_kernel[grid_update](
        key_rot, value, key_cache, value_cache, cache_position,
        B, num_kv_heads, S, H,
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        cache_stride,
        BLOCK_H=64, num_warps=4
    )

    return query_rot, key_rot, key_cache, value_cache


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run_with_triton(*args)


def run(*args):
    return ModelNew()(*args)
