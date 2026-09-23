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

    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr,
                     sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_ptr + pid)
    inv_rms = 1.0 / tl.sqrt(sum_val / H + 1e-6)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = (x_vals * inv_rms) * w_vals
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                               B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                               position_stride,
                               cos_stride0, cos_stride1, cos_stride2,
                               sin_stride0, sin_stride1, sin_stride2,
                               BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * position_stride + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Build emb for first half and mirror for second half
        first_half = idx < H2
        inv_freq_first = tl.load(inv_freq_ptr + (idx), mask=first_half, other=0.0).to(tl.float32)
        inv_freq_second = tl.load(inv_freq_ptr + (idx - H2), mask=~first_half, other=0.0).to(tl.float32)
        inv_freq_vals = tl.where(first_half, inv_freq_first, inv_freq_second)

        emb = pos * inv_freq_vals
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
    h = (pid % (num_heads * S)) // S
    s = pid % S

    half = H // 2
    base = b * cos_stride0 + s * cos_stride1

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # rotated_half = [-x[half:], x[:half]]
        rotated_first = tl.zeros([BLOCK_H], dtype=tl.float32)
        rotated_second = tl.zeros([BLOCK_H], dtype=tl.float32)
        # For idx < half: rotated[idx] = -x[half + idx]
        rotated_first = -tl.load(x_ptr + (b * batch_stride_x + h * h_stride_x + s * s_stride_x + (idx + half)), mask=(idx < half), other=0.0).to(tl.float32)
        # For idx >= half: rotated[idx] = x[idx - half]
        rotated_second = tl.load(x_ptr + (b * batch_stride_x + h * h_stride_x + s * s_stride_x + (idx - half)), mask=(idx >= half), other=0.0).to(tl.float32)
        rotated = tl.where(idx < half, rotated_first, rotated_second)

        y = x_vals * cos_vals + rotated * sin_vals

        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def update_cache_kernel(x_ptr, cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_x, h_stride_x, s_stride_x,
                         batch_stride_cache, head_stride_cache, pos_stride_cache,
                         BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    dest_pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        offs_cache = b * batch_stride_cache + h * head_stride_cache + dest_pos * pos_stride_cache + idx
        tl.store(cache_ptr + offs_cache, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure device compatibility
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda, "All tensors must be on CUDA for Triton kernels."
        B, num_q_heads, S, H = query.shape
        _, num_kv_heads, _, _ = key.shape
        device = query.device

        # 1) RMSNorm for query (bfloat16 output)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        BLOCK_H = 128

        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        query_norm = torch.empty_like(query)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm,
            sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 2) Compute rotation sin/cos per (b, s): cos, sin of shape [B, S, H], float32
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        H2 = H // 2
        rotate_sin_cos_kernel_b_s[(B * S,)](
            position_ids, inv_freq, cos, sin,
            B, S, H, H2,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
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
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 4) RMSNorm for key (bfloat16 output)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        key_norm = torch.empty_like(key)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm,
            sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
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
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        # For key_cache update: copy per (b, kv_head, s) to dest row cache_position[s]
        update_cache_kernel[(B * num_kv_heads * S,)](
            key_rot, key_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # For value_cache: copy per (b, kv_head, s) to


def run(*args):
    return ModelNew()(*args)
