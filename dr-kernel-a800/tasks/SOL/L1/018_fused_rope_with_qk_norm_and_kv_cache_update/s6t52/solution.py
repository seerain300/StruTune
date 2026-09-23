import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    x_stride0, x_stride1, x_stride2,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S
    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid * sum_stride0, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, y_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     x_stride0, x_stride1, x_stride2,
                     y_stride0, y_stride1, y_stride2,
                     w_stride0, sum_stride0,
                     eps: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_sums_ptr + pid * sum_stride0).to(tl.float32)
    H_f = H.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_val / H_f + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + w_stride0 + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = (x_vals * inv_rms) * w_vals
        offs_y = b * y_stride0 + head * y_stride1 + s * y_stride2 + idx
        tl.store(y_ptr + offs_y, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              position_stride0,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)  # 0..B-1
    s = tl.program_id(1)  # 0..S-1
    base = b * position_stride0 + s  # position_ids[b, s] is a scalar int64
    pos = tl.load(position_ids_ptr + base).to(tl.float32)

    H2 = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half = idx < H2

        # Load inv_freq for first half indices only
        inv_freq_idx = idx // 2
        inv_freq_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate for second half: emb[i] = emb[i - H2]
        emb = tl.where(first_half, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base_out = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base_out + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base_out + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           x_stride0, x_stride1, x_stride2,
                           out_stride0, out_stride1, out_stride2,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    # Compute y = x * cos + rotated * sin, rotated mapping as described
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        offs_x = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        base_cs = b * cos_stride0 + s * cos_stride1
        cos_vals = tl.load(cos_ptr + base_cs + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_cs + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        # Build rotated
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # idx < half -> rotated[idx] = -x[half + idx]
        mask_first = idx < half
        # Load relevant x segments
        x_first = tl.load(x_ptr + offs_x, mask=mask_first, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + offs_x, mask=mask_first, other=0.0).to(tl.float32)  # not used here, but keep for clarity
        rotated = rotated + (-x_first)
        # idx >= half -> rotated[idx] = x[idx - half]
        mask_second = ~mask_first
        x_second_part = tl.load(x_ptr + offs_x, mask=mask_second, other=0.0).to(tl.float32)
        rotated = rotated + x_second_part

        y_vals = x_vals * cos_vals + rotated * sin_vals
        offs_out = b * out_stride0 + head * out_stride1 + s * out_stride2 + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                         H: tl.constexpr,
                         query_rot_stride0, query_rot_stride1, query_rot_stride2,
                         value_stride0, value_stride1, value_stride2,
                         key_cache_stride0, key_cache_stride1, key_cache_stride2,
                         value_cache_stride0, value_cache_stride1, value_cache_stride2,
                         BLOCK_H: tl.constexpr):
    b = tl.program_id(0)  # 0..B-1
    kv = tl.program_id(1)  # 0..num_kv_heads-1
    s = tl.program_id(2)  # 0..S-1
    dest_pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    # Copy rotated query into key_cache at dest_pos
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_in = b * query_rot_stride0 + kv * query_rot_stride1 + s * query_rot_stride2 + idx
        vals = tl.load(query_rot_ptr + offs_in, mask=mask, other=0.0).to(tl.float32)
        offs_out_k = b * key_cache_stride0 + kv * key_cache_stride1 + dest_pos * key_cache_stride2 + idx
        tl.store(key_cache_ptr + offs_out_k, vals, mask=mask)

    # Copy value into value_cache at dest_pos
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_in = b * value_stride0 + kv * value_stride1 + s * value_stride2 + idx
        vals = tl.load(value_ptr + offs_in, mask=mask, other=0.0).to(tl.float32)
        offs_out_v = b * value_cache_stride0 + kv * value_cache_stride1 + dest_pos * value_cache_stride2 + idx
        tl.store(value_cache_ptr + offs_out_v, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        device = query.device
        dtype = query.dtype
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda \
            and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        BLOCK_H = 128
        grid_sum_q = (B * num_q_heads * S,)
        # 1) RMSNorm sum for query
        sum_q = torch.empty((grid_sum_q[0],), dtype=torch.float32, device=device)
        rms_sum_kernel[grid_sum_q](
            query, sum_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_q.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        # 2) RMSNorm for query -> query_norm (bfloat16 output)
        query_norm = torch.empty_like(query, dtype=torch.float32, device=device)
        grid_rms_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_rms_q](
            query, q_norm_weight, query_norm, sum_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            k_norm_weight.stride(0), sum_q.stride(0),
            rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4
        )
        # Convert query_norm to bfloat16 for rotation
        query_norm_bf16 = query_norm.to(torch.bfloat16)

        # 3) Compute rotation sin/cos for each (b, s): cos[B, S, H], sin[B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        H2 = H // 2
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H, H2,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 4) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.bfloat16, device=device)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 5) RMSNorm sum for key
        sum_key = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_key.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        # 6) RMSNorm for key -> key_norm (compute in float32, output bfloat16)
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=device)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0), sum_key.stride(0),
            rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4
        )
        # Convert to bfloat16 for rotation
        key_norm_bf16 = key_norm.to(torch.bfloat16)

        # 7) Compute rotation sin/cos for each (b, s) (reuse cos/sin already computed for query)
        # They depend only on position_ids and inv_freq; computed once above.

        # 8) Apply rotation to key_norm -> key_rot (bfloat16 output)
        key_rot = torch.empty((B, num_kv_heads, S, H), dtype=torch.bfloat16, device=device)
        apply_rotation_kernel[(B * num_kv_heads * S,)](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 9) Update caches:
        # key_cache[:, :, cache_position] = rotated_query
        # value_cache[:, :, cache_position] = value
        # Note: cache_position is int64 vector of length S. Triton expects int32 offsets, so cast.
        cache_pos_i32 = cache_position.to(torch.int32)
        update_cache_kernel[(B, num_kv_heads, S)](
            query_rot, value, key_cache, value_cache,
            cache_pos_i32,
            B, num_kv_heads, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
