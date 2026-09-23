import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # Each program handles one (b, head, s) token and reduces across H
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    eps: tl.float32,
                    batch_stride_x, h_stride_x, s_stride_x,
                    batch_stride_out, h_stride_out, s_stride_out,
                    BLOCK_H: tl.constexpr):
    # Each program applies normalization to one (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_sums_ptr + pid)
    Hf = tl.full((), H, tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_sq / Hf + eps)
    # Scale by weight
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals
        out_offs = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              position_stride,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    pos = tl.load(position_ids_ptr + b * position_stride + s)

    # Build emb of length H: for i < H//2, emb[i] = pos * inv_freq[i//2]; for i >= H//2, emb[i] = emb[i - H//2]
    half = H // 2
    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # First half
        first_mask = idx < half
        inv_freq_idx = (idx // 2).to(tl.int32)
        inv_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_vals
        # Second half: reuse values from first half
        second_mask = idx >= half
        # For idx >= half, we map to emb[idx - half] = emb_first[idx - half]
        # idx - half ranges [0, half-1] when idx in [half, 2*half-1]
        idx_minus_half = idx - half
        emb_second = tl.load(inv_freq_ptr + (idx_minus_half // 2), mask=second_mask, other=0.0).to(tl.float32) * pos
        emb = tl.where(first_mask, emb_first, emb_second)

        c = tl.cos(emb)
        s_ = tl.sin(emb)
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
    head = (pid % (num_heads * S)) // S
    s = pid % S

    base_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out
    base_cos = b * cos_stride0 + s * cos_stride1

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_cos + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Construct rotated part
        half = H // 2
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # For idx < half: rotated[idx] = x[half + idx]
        first_mask = idx < half
        x_first = tl.load(x_ptr + x_offs, mask=first_mask, other=0.0).to(tl.float32)
        rotated = tl.where(first_mask, x_first, rotated)
        # For idx >= half: rotated[idx] = x[idx - half]
        second_mask = idx >= half
        x_second = tl.load(x_ptr + (b * batch_stride_x + head * h_stride_x + s * s_stride_x + (idx - half)), mask=second_mask, other=0.0).to(tl.float32)
        rotated = tl.where(second_mask, x_second, rotated)

        y = x_vals * cos_vals + rotated * sin_vals
        tl.store(out_ptr + base_out + idx, y, mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_position_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         query_batch_stride, query_h_stride, query_s_stride,
                         key_cache_batch_stride, key_cache_h_stride, key_cache_s_stride,
                         value_cache_batch_stride, value_cache_h_stride, value_cache_s_stride,
                         cache_pos_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_head = (pid % (num_kv_heads * S)) // S
    s = pid % S

    # Read rotated query for token s: shape [H]
    sum_offs = 0
    # We need to sum H elements; do it in blocks to compute base idx
    # However, we can simply read per element with masking using BLOCK_H
    base_q = b * query_batch_stride + kv_head * query_h_stride + s * query_s_stride
    dest_pos = tl.load(cache_position_ptr + s * cache_pos_stride)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        q_offs = base_q + idx
        rotated_vals = tl.load(query_rot_ptr + q_offs, mask=mask, other=0.0).to(tl.float32)

        # Write to key_cache at row dest_pos
        base_k = b * key_cache_batch_stride + kv_head * key_cache_h_stride + dest_pos * key_cache_s_stride
        tl.store(key_cache_ptr + base_k + idx, rotated_vals, mask=mask)

        # Write value (original, not rotated) to value_cache at row dest_pos
        base_v = b * value_cache_batch_stride + kv_head * value_cache_h_stride + dest_pos * value_cache_s_stride
        v_offs = b * query_batch_stride + kv_head * query_h_stride + s * query_s_stride + idx  # same layout as value_ptr: [B, num_kv_heads, S, H]
        v_vals = tl.load(value_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + base_v + idx, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
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
        # Ensure tensors are on the same device (CUDA) and bfloat16 dtype for query/key/value
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda and \
               key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda and \
               q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda, "All tensors must be on CUDA."
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16, "query/key/value must be bfloat16."
        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        H = query.shape[3]
        num_kv_heads = key.shape[1]
        # 1) RMS sum for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 2) RMSNorm for query -> query_norm (bfloat16)
        query_norm = torch.empty_like(query)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            float(rms_norm_eps),
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 3) Compute rotation cos/sin per (b, s) in float32 -> [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 4) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 5) RMS sum for key
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 6) RMSNorm for key -> key_norm (bfloat16)
        key_norm = torch.empty_like(key)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm, sum_sums_k,
            B, num_kv_heads, S, H,
            float(rms_norm_eps),
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 7) Apply rotation to key_norm -> key_rot (bfloat16) [for completeness if needed by caller]
        key_rot = torch.empty_like(key_norm)
        apply_rotation_kernel[(B * num_kv_heads * S,)](
            key_norm, cos, sin, key_rot,
            B, num_kv_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 8) Update caches using rotated query and original value at cache_position
        update_cache_kernel[(B * num_kv_heads * S,)](
            query_rot, value, key_cache, value_cache,
            cache_position,
            B, num_kv_heads, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position.stride(0),
            BLOCK_H=64,
            num_warps=4
        )

        # Return as original function: rotated query, rotated key, updated caches
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
