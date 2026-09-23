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

    # Load sum and compute inv_rms
    out_off = b * sum_stride0 + h * sum_stride1 + S * sum_stride2
    sum_scalar = tl.load(sum_ptr + out_off).to(tl.float32)
    mean = sum_scalar / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Apply weight and write normalized output
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx
        offs_out = b * batch_stride_out + h * head_stride_out + s * s_stride_out + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx * weight_stride, mask=mask, other=1.0).to(tl.float32)
        out_vals = x_vals * inv_rms * w_vals
        tl.store(out_ptr + offs_out, out_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              pos_stride,  # stride of position_ids (assumed contiguous: 1)
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
        half = H // 2
        first_mask = idx < half
        inv_idx = idx // 2  # i -> inv_freq[i//2]
        emb_first = pos * tl.load(inv_freq_ptr + inv_idx, mask=first_mask, other=0.0).to(tl.float32)
        # duplicate second half
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

    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * s_stride_x + idx

        # Load x
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for (b, s)
        base = b * cos_stride0 + s * cos_stride1
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated according to standard RoPE: rotated[i] = -x[half+i] for i<half; rotated[half+i] = x[i]
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # idx < half: rotated[idx] = -x[half + idx]
        first_mask = idx < half
        rotated_first = -tl.load(x_ptr + (b * batch_stride_x + h * head_stride_x + s * s_stride_x + (half + idx)), mask=first_mask, other=0.0).to(tl.float32)
        rotated = tl.where(first_mask, rotated_first, rotated)
        # idx >= half: rotated[idx] = x[idx - half]
        second_mask = ~first_mask
        rotated_second = tl.load(x_ptr + (b * batch_stride_x + h * head_stride_x + s * s_stride_x + (idx - half)), mask=second_mask, other=0.0).to(tl.float32)
        rotated = tl.where(second_mask, rotated_second, rotated)

        y = x_vals * cos_vals + rotated * sin_vals

        # Store result
        offs_out = b * batch_stride_out + h * head_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def update_cache_kernel(rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_rot, head_stride_rot, s_stride_rot,
                         batch_stride_keyc, kv_head_stride_keyc, pos_stride_keyc,
                         batch_stride_valc, kv_head_stride_valc, pos_stride_valc,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest_pos = tl.load(cache_pos_ptr + s).to(tl.int32)
    half = H // 2

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load rotated row (rot_ptr has shape [B, num_q_heads, S, H]; we pass rotated key)
        offs_rot = b * batch_stride_rot + kv_h * head_stride_rot + s * s_stride_rot + idx
        rot_vals = tl.load(rot_ptr + offs_rot, mask=mask, other=0.0).to(tl.float32)

        # Store to key_cache[:, :, cache_position[s]]
        offs_keyc = b * batch_stride_keyc + kv_h * kv_head_stride_keyc + dest_pos * pos_stride_keyc + idx
        tl.store(key_cache_ptr + offs_keyc, rot_vals, mask=mask)

        # Store to value_cache[:, :, cache_position[s]] = value
        offs_valc = b * batch_stride_valc + kv_h * kv_head_stride_valc + dest_pos * pos_stride_valc + idx
        val_vals = tl.load(value_ptr + (b * value_ptr.stride(0) + kv_h * value_ptr.stride(1) + s * value_ptr.stride(2) + idx), mask=mask, other=0.0).to(tl.float32)  # dummy but Triton requires loads; we can use rot_vals
        tl.store(value_cache_ptr + offs_valc, val_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        query: [B, num_q_heads, S, H]
        key: [B, num_kv_heads, S, H]
        value: [B, num_kv_heads, S, H] (original code uses bfloat16)
        position_ids: [B, S] (int64)
        key_cache: [B, num_kv_heads, max_pos, H]
        value_cache: [B, num_kv_heads, max_pos, H]
        cache_position: [S] (int64)
        q_norm_weight, k_norm_weight: [H] (bfloat16, original uses ones)
        inv_freq: [(H//2)] (float32)
        rms_norm_eps: float
        """

        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA."
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        device = query.device

        # 1) RMSNorm for query -> query_norm (float32 output)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        BLOCK_H = 64
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        query_norm = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=device)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight.to(torch.float32), query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            q_norm_weight.stride(0),
            sum_sums_q.stride(0), sum_sums_q.stride(1), sum_sums_q.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 2) Build cos/sin per (b, s) of shape [B, S, H] (float32)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids.to(torch.int64), inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16 output)
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

        # 4) RMSNorm for key -> key_norm (float32 output)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            BLOCK_H=BLOCK_H, num_warps=4
        )
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=device)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight.to(torch.float32), key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0),
            sum_sums_k.stride(0), sum_sums_k.stride(1), sum_sums_k.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=BLOCK_H, num_warps=4
        )

        # 5) Apply rotation to key_norm -> key_rot (bfloat16 output)
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

        # 6) Update caches: key_cache[:,


def run(*args):
    return ModelNew()(*args)
