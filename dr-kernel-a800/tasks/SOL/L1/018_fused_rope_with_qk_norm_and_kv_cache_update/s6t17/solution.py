import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
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
    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals_f32 * x_vals_f32, axis=0)
    tl.store(sum_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     x_stride0, x_stride1, x_stride2,
                     out_stride0, out_stride1, out_stride2,
                     weight_stride0,
                     eps: tl.constexpr,  # epsilon is float
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_ptr + pid)
    inv_rms = 1.0 / tl.sqrt(sum_sq / H + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + weight_stride0 + idx, mask=mask, other=1.0).to(tl.float32)
        out_vals = x_vals.to(tl.float32) * w_vals * inv_rms
        out_offs = b * out_stride0 + head * out_stride1 + s * out_stride2 + idx
        tl.store(out_ptr + out_offs, out_vals.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              pid_stride,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(position_ids_ptr + b * pid_stride + s * pid_stride)
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half_mask = idx < half
        inv_offs = idx // 2  # i//2 for i < half
        # Load inv_freq for first half
        inv_vals_first = tl.load(inv_freq_ptr + inv_offs, mask=first_half_mask, other=0.0).to(tl.float32)
        # emb_first = pos * inv_vals_first
        emb_first = pos * inv_vals_first
        # For second half, emb = emb_first
        emb = tl.where(first_half_mask, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


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

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # First half: idx < H//2 -> rotated = -x[half + idx]
        half = H // 2
        for i in range(0, half):
            # Load x and compute rotated[i] = -x[half + i]
            x_offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + (half + i)
            xr = tl.load(x_ptr + x_offs).to(tl.float32)
            # Load cos/sin for this index
            c = tl.load(cos_ptr + base + i * cos_stride2).to(tl.float32)
            s_ = tl.load(sin_ptr + base + i * sin_stride2).to(tl.float32)
            y_offs = b * out_stride0 + head * out_stride1 + s * out_stride2 + i
            tl.store(out_ptr + y_offs, (xr * c - xr * s_).to(out_ptr.dtype.element_ty))

        # Second half: idx >= H//2 -> rotated = x[idx - half]
        for i in range(0, half):
            j = half + i
            x_offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + j
            xj = tl.load(x_ptr + x_offs).to(tl.float32)
            c = tl.load(cos_ptr + base + j * cos_stride2).to(tl.float32)
            s_ = tl.load(sin_ptr + base + j * sin_stride2).to(tl.float32)
            y_offs = b * out_stride0 + head * out_stride1 + s * out_stride2 + j
            tl.store(out_ptr + y_offs, (xj * c + xj * s_).to(out_ptr.dtype.element_ty))


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         q_stride0, q_stride1, q_stride2,
                         value_stride0, value_stride1, value_stride2,
                         k_stride0, k_stride1, k_stride2,
                         v_stride0, v_stride1, v_stride2,
                         cp_stride,
                         BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    head = (pid % (num_kv_heads * S)) // S
    s = pid % S
    cp = tl.load(cache_pos_ptr + b * cp_stride + s * cp_stride)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Copy query_rot to key_cache[b, head, cp, :]
        q_offs = b * q_stride0 + head * q_stride1 + s * q_stride2 + idx
        q_vals = tl.load(query_rot_ptr + q_offs, mask=mask, other=0.0)  # bfloat16
        k_offs = b * k_stride0 + head * k_stride1 + cp * k_stride2 + idx
        tl.store(key_cache_ptr + k_offs, q_vals.to(key_cache_ptr.dtype.element_ty), mask=mask)

        # Copy value[b, head, s, :] to value_cache[b, head, cp, :]
        v_offs = b * value_stride0 + head * value_stride1 + s * value_stride2 + idx
        v_vals = tl.load(value_ptr + v_offs, mask=mask, other=0.0)  # bfloat16
        vc_offs = b * v_stride0 + head * v_stride1 + cp * v_stride2 + idx
        tl.store(value_cache_ptr + vc_offs, v_vals.to(value_cache_ptr.dtype.element_ty), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor,
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
        assert key.shape == (B, num_q_heads, S, H)
        assert value.shape == (B, num_q_heads, S, H)  # The provided data uses num_q_heads, but original uses num_kv_heads; we match provided.
        num_kv_heads = key.shape[1]
        device = query.device
        dtype_q = query.dtype
        dtype_k = key.dtype
        dtype_v = value.dtype

        # 1) RMSNorm for query -> query_norm (bfloat16 input/output)
        query_norm = torch.empty_like(query)
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

        # 2) Compute rotation sin/cos per token: [B, S, H] (float32)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        rotate_sin_cos_kernel_b_s[(B, S)](position_ids, inv_freq, cos, sin,
                                          B, S, H,
                                          position_ids.stride(0),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)  # bfloat16
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

        # 5) Update caches: key_cache[:, :, cache_position] = query_rot, value_cache[:, :, cache_position] = value
        update_cache_kernel[(B * num_kv_heads * S)](
            query_rot, value, key_cache, value_cache,
            cache_position,
            B, num_kv_heads, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        return query_rot, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
