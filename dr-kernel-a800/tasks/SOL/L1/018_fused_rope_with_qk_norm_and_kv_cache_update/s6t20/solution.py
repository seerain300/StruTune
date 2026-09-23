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
    # Store sum (one scalar per (b, head, s))
    out_off = b * sum_stride0 + h * sum_stride1 + s * sum_stride2
    tl.store(sum_ptr + out_off, total)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, s_stride_x,
                    batch_stride_out, head_stride_out, s_stride_out,
                    weight_stride,  # typically 1 for [H]
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

    # Load sum of squares and compute inv_rms
    sum_val = tl.load(sum_ptr + b * sum_stride0 + h * sum_stride1 + s * sum_stride2).to(tl.float32)
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)  # float32

    # Normalize and scale by weight
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals
        tl.store(out_ptr + b * batch_stride_out + h * head_stride_out + s * s_stride_out + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                              pos_stride,  # usually 1
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        first_half = idx < H2
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_half, other=0.0).to(tl.float32)  # idx in [0, H2)
        emb_first = pos * inv_freq_vals  # shape [BLOCK_H]
        emb = tl.where(first_half, emb_first, emb_first)  # duplicate to second half
        # emb is valid only for idx < H; second half elements are duplicated correctly
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

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        offs_x = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros_like(x_vals)
        # Map for rotation:
        # For i < half: rotated[i] = -x[half + i]
        # For i >= half: rotated[i] = x[i - half]
        rotated_part1 = -tl.load(x_ptr + offs_x, mask=(idx < half), other=0.0).to(tl.float32)
        rotated_part2 = tl.load(x_ptr + offs_x, mask=(idx >= half), other=0.0).to(tl.float32)
        rotated = rotated_part1 + rotated_part2  # rotated_part2 corresponds to i >= half

        y = x_vals * cos_vals + rotated * sin_vals
        tl.store(out_ptr + b * batch_stride_out + h * head_stride_out + s * s_stride_out + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_key, head_stride_key, s_stride_key,
                         batch_stride_value, head_stride_value, s_stride_value,
                         cache_bs_stride, cache_hs_stride, cache_d_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest = tl.load(cache_pos_ptr + s).to(tl.int64)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load key_rot[b, h, s, :]
        offs_k = b * batch_stride_key + h * head_stride_key + s * s_stride_key + idx
        k_vals = tl.load(key_rot_ptr + offs_k, mask=mask, other=0.0).to(tl.bfloat16)

        # Store to key_cache[b, h, dest, :]
        offs_ck = b * cache_bs_stride + h * cache_hs_stride + dest * cache_d_stride + idx
        tl.store(key_cache_ptr + offs_ck, k_vals, mask=mask)

        # Load value[b, h, s, :]
        offs_v = b * batch_stride_value + h * head_stride_value + s * s_stride_value + idx
        v_vals = tl.load(value_ptr + offs_v, mask=mask, other=0.0).to(tl.bfloat16)

        # Store to value_cache[b, h, dest, :]
        offs_cv = b * cache_bs_stride + h * cache_hs_stride + dest * cache_d_stride + idx
        tl.store(value_cache_ptr + offs_cv, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        # Shapes
        B, num_q_heads, S, H = query.shape
        Bk, num_kv_heads, Sk, Hk = key.shape  # Sk should be S
        assert Sk == S and Hk == H, "key/value shapes must match query on S and head_dim"
        assert key_cache.shape[0] == B and value_cache.shape[0] == B
        assert key_cache.shape[1] == num_kv_heads and value_cache.shape[1] == num_kv_heads
        assert key_cache.shape[2] == key_cache.shape[2], "key_cache dim2 must be valid"
        assert value_cache.shape[2] == value_cache.shape[2], "value_cache dim2 must be valid"
        assert cache_position.shape[0] == S, "cache_position must have length S"

        # Ensure bfloat16
        query = query.contiguous().to(torch.bfloat16)
        key = key.contiguous().to(torch.bfloat16)
        value = value.contiguous().to(torch.bfloat16)

        # 1) RMSNorm for query -> query_norm (bfloat16)
        query_norm = torch.empty_like(query)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        BLOCK_H = 128
        grid_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_q](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        rms_norm_kernel[grid_q](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            q_norm_weight.stride(0),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 2) Compute sin/cos for rotation: [B, S, H], float32
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](
            position_ids, inv_freq, cos, sin,
            B, S, H, H2,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (bfloat16)
        key_norm = torch.empty_like(key)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        rms_norm_kernel[grid_k](
            key, k_norm_weight, key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot (bfloat16)
        key_rot = torch.empty_like(key_norm)
        apply_rotation_kernel[grid_k](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        # Ensure cache tensors are contiguous
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        update_cache_kernel[grid_k](
            key_rot, value, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            num_warps=4, BLOCK_H=BLOCK_H
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
