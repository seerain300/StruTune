import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B, num_heads, S, H,
                    x_stride0, x_stride1, x_stride2,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    s = pid % S
    # sum over head dim
    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_stride0 + s * x_stride2 + idx * x_stride2
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x32 = x_vals.to(tl.float32)
        sum_val += tl.sum(x32 * x32, axis=0)
    tl.store(sum_sums_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                     B, num_heads, S, H,
                     x_stride0, x_stride1, x_stride2,
                     out_stride0, out_stride1, out_stride2,
                     weight_stride0,
                     eps: tl.float32,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    s = pid % S
    sum_val = tl.load(sum_sums_ptr + pid)
    inv_rms = 1.0 / tl.sqrt(sum_val / H + eps)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * x_stride0 + s * x_stride2 + idx * x_stride2
        offs_out = b * out_stride0 + s * out_stride2 + idx * out_stride2
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x_vals * (w_vals * inv_rms)
        tl.store(out_ptr + offs_out, y.to(tl.float32), mask=mask)  # output kept in float32 for now


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr,
                              cos_ptr, sin_ptr,
                              B, S, H,
                              pos_stride0,  # position_ids.stride(0)
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(position_ids_ptr + b * pos_stride0 + s).to(tl.float32)
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # emb first half: pos * inv_freq[:half]
        first_mask = idx < half
        inv_freq_idx = idx // 2  # since H is even, idx < half -> idx//2 < half
        emb_first = pos * tl.load(inv_freq_ptr + inv_freq_idx, mask=first_mask, other=0.0).to(tl.float32)
        # emb second half is duplicated
        emb = tl.where(first_mask, emb_first, emb_first)
        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B, num_heads, S, H,
                           x_stride0, x_stride1, x_stride2,
                           out_stride0, out_stride1, out_stride2,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S
    # First half
    half = H // 2
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        x_offs = b * x_stride0 + h * x_stride1 + s * x_stride2 + idx
        out_offs = b * out_stride0 + h * out_stride1 + s * out_stride2 + idx
        x_first = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        cos_vals = tl.load(cos_ptr + b * cos_stride0 + s * cos_stride1 + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + b * sin_stride0 + s * sin_stride1 + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)
        rotated_first = -tl.load(x_ptr + (b * x_stride0 + h * x_stride1 + s * x_stride2 + (idx + half)), mask=mask, other=0.0).to(tl.float32)
        y = x_first * cos_vals + rotated_first * sin_vals
        tl.store(out_ptr + out_offs, y, mask=mask)  # out_ptr dtype controls casting
    # Second half
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        x_offs = b * x_stride0 + h * x_stride1 + s * x_stride2 + (idx + half)
        out_offs = b * out_stride0 + h * out_stride1 + s * out_stride2 + (idx + half)
        x_second = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        cos_vals = tl.load(cos_ptr + b * cos_stride0 + s * cos_stride1 + (idx + half) * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + b * sin_stride0 + s * sin_stride1 + (idx + half) * sin_stride2, mask=mask, other=0.0).to(tl.float32)
        rotated_second = tl.load(x_ptr + (b * x_stride0 + h * x_stride1 + s * x_stride2 + idx), mask=mask, other=0.0).to(tl.float32)
        y = x_second * cos_vals + rotated_second * sin_vals
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_position_ptr,
                         B, num_kv_heads, S, H,
                         query_stride0, query_stride1, query_stride2,
                         value_stride0, value_stride1, value_stride2,
                         key_stride0, key_stride1, key_stride2,
                         val_stride0, val_stride1, val_stride2,
                         cp_stride0):
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    h = (pid % (num_kv_heads * S)) // S
    s = pid % S
    pos = tl.load(cache_position_ptr + s * cp_stride0)
    # write rotated query to key_cache
    for off in range(0, H, 128):
        idx = off + tl.arange(0, 128)
        mask = idx < H
        src_offs = b * query_stride0 + h * query_stride1 + s * query_stride2 + idx
        dest_offs = b * key_stride0 + h * key_stride1 + pos * key_stride2 + idx
        val = tl.load(query_rot_ptr + src_offs, mask=mask, other=0.0)  # query_rot is float32
        tl.store(key_cache_ptr + dest_offs, val, mask=mask)
    # write value to value_cache
    for off in range(0, H, 128):
        idx = off + tl.arange(0, 128)
        mask = idx < H
        src_offs = b * value_stride0 + h * value_stride1 + s * value_stride2 + idx
        dest_offs = b * val_stride0 + h * val_stride1 + pos * val_stride2 + idx
        val = tl.load(value_ptr + src_offs, mask=mask, other=0.0)  # value is bfloat16
        tl.store(value_cache_ptr + dest_offs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        Bk, num_kv_heads, Sk, Hk = key.shape  # Sk should equal S
        assert B == Bk and Sk == S and H == Hk, "Shape mismatch between query, key, value"
        device = query.device

        # 1) RMSNorm for query -> query_norm (float32 output to avoid precision loss)
        query_norm = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=device)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        BLOCK_H = 128
        grid_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               sum_sums_q.stride(0), BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_q](query, q_norm_weight, query_norm, sum_sums_q,
                                B, num_q_heads, S, H,
                                query.stride(0), query.stride(1), query.stride(2),
                                query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                q_norm_weight.stride(0),
                                rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 2) Rotation sin/cos per token: [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        rotate_sin_cos_kernel_b_s[(B, S)](position_ids, inv_freq, cos, sin,
                                          B, S, H,
                                          position_ids.stride(0),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.bfloat16, device=device)
        apply_rotation_kernel[grid_q](query_norm, cos, sin, query_rot,
                                      B, num_q_heads, S, H,
                                      query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                      query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                      cos.stride(0), cos.stride(1), cos.stride(2),
                                      sin.stride(0), sin.stride(1), sin.stride(2),
                                      BLOCK_H=BLOCK_H, num_warps=4)

        # 4) RMSNorm for key -> key_norm (float32 output)
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=device)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](key, sum_sums_k,
                               B, num_kv_heads, S, H,
                               key.stride(0), key.stride(1), key.stride(2),
                               sum_sums_k.stride(0), BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_k](key, k_norm_weight, key_norm, sum_sums_k,
                                B, num_kv_heads, S, H,
                                key.stride(0), key.stride(1), key.stride(2),
                                key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                k_norm_weight.stride(0),
                                rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 5) Update caches: key_cache[:, :, cache_position] = query_rot
        #    value_cache[:, :, cache_position] = value


def run(*args):
    return ModelNew()(*args)
