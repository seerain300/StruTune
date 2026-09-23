import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    x_batch_stride, x_head_stride, x_s_stride,
                    sum_stride, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    total = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_batch_stride + h * x_head_stride + s * x_s_stride + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x_vals * x_vals, axis=0)
    inv_H = 1.0 / H
    sum_sums_ptr[pid] = total * inv_H


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr,
                     sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     x_batch_stride, x_head_stride, x_s_stride,
                     out_batch_stride, out_head_stride, out_s_stride,
                     w_stride,
                     rms_norm_eps: tl.constexpr,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    inv_rms = 1.0 / tl.sqrt(sum_sums_ptr[pid] + rms_norm_eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_batch_stride + h * x_head_stride + s * x_s_stride + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        w_vals = tl.load(weight_ptr + w_stride * idx, mask=mask, other=0.0).to(tl.float32)

        y = x_vals * w_vals * inv_rms
        out_offs = b * out_batch_stride + h * out_head_stride + s * out_s_stride + idx
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              position_stride,
                              cos_batch_stride, cos_s_stride, cos_h_stride,
                              sin_batch_stride, sin_s_stride, sin_h_stride,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if (b >= B) or (s >= S):
        return

    pos = tl.load(position_ids_ptr + b * position_stride + s * 0)  # int64
    pos = pos.to(tl.float32)

    # Build emb of length H: first half is pos * inv_freq[:H//2], second half duplicates
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half_mask = idx < half

        # inv_freq has length half
        inv_freq_vals = tl.load(inv_freq_ptr + idx * 0, mask=first_half_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals

        # emb for full H: duplicate first half to second half
        emb = tl.where(first_half_mask, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_batch_stride + s * cos_s_stride
        tl.store(cos_ptr + base + idx * cos_h_stride, c, mask=mask)
        tl.store(sin_ptr + (b * sin_batch_stride + s * sin_s_stride) + idx * sin_h_stride, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           x_batch_stride, x_head_stride, x_s_stride,
                           out_batch_stride, out_head_stride, out_s_stride,
                           cos_batch_stride, cos_s_stride, cos_h_stride,
                           sin_batch_stride, sin_s_stride, sin_h_stride,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_batch_stride + s * cos_s_stride
    half = H // 2

    # First half: idx < half
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        offs_x = b * x_batch_stride + h * x_head_stride + s * x_s_stride + idx
        cos_offs = base + idx * cos_h_stride
        sin_offs = (b * sin_batch_stride + s * sin_s_stride) + idx * sin_h_stride

        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        cos_vals = tl.load(cos_ptr + cos_offs, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + sin_offs, mask=mask, other=0.0).to(tl.float32)

        rotated_half = tl.load(x_ptr + (b * x_batch_stride + h * x_head_stride + s * x_s_stride + (idx + half)), mask=mask, other=0.0).to(tl.float32)  # x[half + idx]
        rotated_half = -rotated_half  # rotate_half takes negative of second half

        y = x_vals * cos_vals + rotated_half * sin_vals
        out_offs = b * out_batch_stride + h * out_head_stride + s * out_s_stride + idx
        tl.store(out_ptr + out_offs, y, mask=mask)

    # Second half: idx >= half
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = (idx + half) < H  # equivalent to idx < half
        offs_x = b * x_batch_stride + h * x_head_stride + s * x_s_stride + (idx + half)
        cos_offs = base + (idx + half) * cos_h_stride
        sin_offs = (b * sin_batch_stride + s * sin_s_stride) + (idx + half) * sin_h_stride

        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        cos_vals = tl.load(cos_ptr + cos_offs, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + sin_offs, mask=mask, other=0.0).to(tl.float32)

        rotated_half = tl.load(x_ptr + (b * x_batch_stride + h * x_head_stride + s * x_s_stride + idx), mask=mask, other=0.0).to(tl.float32)  # x[idx] from first half
        rotated_half = -rotated_half  # negative because second half index

        y = x_vals * cos_vals + rotated_half * sin_vals
        out_offs = b * out_batch_stride + h * out_head_stride + s * out_s_stride + (idx + half)
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(rotated_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         rotated_batch_stride, rotated_head_stride, rotated_s_stride,
                         value_batch_stride, value_head_stride, value_s_stride,
                         key_batch_stride, key_head_stride, key_pos_stride, key_h_stride,
                         val_batch_stride, val_head_stride, val_pos_stride, val_h_stride,
                         BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    pos = tl.load(cache_pos_ptr + s)

    # Copy rotated query -> key_cache
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        src_offs = b * rotated_batch_stride + h * rotated_head_stride + s * rotated_s_stride + idx
        dst_offs = b * key_batch_stride + h * key_head_stride + pos * key_pos_stride + idx
        x_vals = tl.load(rotated_ptr + src_offs, mask=mask, other=0.0)
        tl.store(key_cache_ptr + dst_offs, x_vals, mask=mask)

    # Copy value -> value_cache
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        src_offs = b * value_batch_stride + h * value_head_stride + s * value_s_stride + idx
        dst_offs = b * val_batch_stride + h * val_head_stride + pos * val_pos_stride + idx
        v_vals = tl.load(value_ptr + src_offs, mask=mask, other=0.0)
        tl.store(value_cache_ptr + dst_offs, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self,
                query: torch.Tensor,
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
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query -> query_norm
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid_q = (B * num_q_heads * S,)
        BLOCK_H = 128
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               sum_sums_q.stride(0), BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_q](query, q_norm_weight, query_norm, sum_sums_q,
                                B, num_q_heads, S, H,
                                query.stride(0), query.stride(1), query.stride(2),
                                query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                q_norm_weight.stride(0), rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 2) RMSNorm for key -> key_norm
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](key, sum_sums_k,
                               B, num_kv_heads, S, H,
                               key.stride(0), key.stride(1), key.stride(2),
                               sum_sums_k.stride(0), BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_k](key, k_norm_weight, key_norm, sum_sums_k,
                                B, num_kv_heads, S, H,
                                key.stride(0), key.stride(1), key.stride(2),
                                key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                k_norm_weight.stride(0), rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Compute rotation sin/cos per token: [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](position_ids, inv_freq, cos, sin,
                                          B, S, H,
                                          position_ids.stride(0),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 4) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[grid_q](query_norm, cos, sin, query_rot,
                                      B, num_q_heads, S, H,
                                      query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                      query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                      cos.stride(0), cos.stride(1), cos.stride(2),
                                      sin.stride(0), sin.stride(1), sin.stride(2),
                                      BLOCK_H=BLOCK_H, num_warps=4)

        # 5) Update caches: key_cache[:, :, cache_position] = query_rot, value_cache[:, :, cache_position] = value
        # cache_position: [S], int64
        update_cache_kernel[(B * num_kv_heads * S)](query_rot, value, key_cache, value_cache,
                                                    cache_position,
                                                    B, num_kv_heads, S, H,
                                                    query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                                    value.stride(0), value.stride(1), value.stride(2),
                                                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                                                    key_cache.stride(3),
                                                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                                                    value_cache.stride(3),
                                                    BLOCK_H=BLOCK_H, num_warps=4)

        return query_rot, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
