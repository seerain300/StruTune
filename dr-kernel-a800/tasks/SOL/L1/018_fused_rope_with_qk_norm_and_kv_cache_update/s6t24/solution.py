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
    # Reduce across H in chunks
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    # Store partial sum (per token) to sum_sums[pid]
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
    # Scale by weight (elementwise per dim)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        weight_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        out_vals = x_vals * inv_rms * weight_vals
        offs_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, out_vals.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              pos_stride, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    if pid_b >= B or pid_s >= S:
        return
    pos = tl.load(position_ids_ptr + pid_b * pos_stride + pid_s).to(tl.float32)

    H2 = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half = idx < H2

        inv_freq_idx = idx // 2  # valid only for first half
        inv_freq_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals  # shape [BLOCK_H]
        # Duplicate for second half
        emb = tl.where(first_half, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = pid_b * cos_stride0 + pid_s * cos_stride1
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

    base_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x
    base_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out

    H2 = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load original x
        x = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        base = b * cos_stride0 + s * cos_stride1
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated part: rotated[i] = -x[H2 + i] for i < H2; and rotated[H2 + i] = x[i]
        rotated_first = tl.load(x_ptr + base_x + (H2 + idx), mask=(idx < H2), other=0.0).to(tl.float32) * (-1.0)
        rotated_second = tl.load(x_ptr + base_x + (idx - H2), mask=(idx >= H2), other=0.0).to(tl.float32)
        rotated = tl.where(idx < H2, rotated_first, rotated_second)

        y = x * cos_vals + rotated * sin_vals

        tl.store(out_ptr + base_out + idx, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def update_cache_kernel(rot_query_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                         pos_stride, kc_b_stride, kc_h_stride, kc_s_stride,
                         vc_b_stride, vc_h_stride, vc_s_stride,
                         BLOCK_H: tl.constexpr):
    # Each program handles one (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_head = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest = tl.load(pos_stride + s * pos_stride)  # cache_position[s]

    H = rot_query_ptr.shape[-1]
    base_rq = b * rot_query_ptr.stride(0) + kv_head * rot_query_ptr.stride(1) + s * rot_query_ptr.stride(2)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        vals = tl.load(rot_query_ptr + base_rq + idx, mask=mask, other=0.0).to(tl.float32)
        base_kc = b * kc_b_stride + kv_head * kc_h_stride + dest * kc_s_stride
        tl.store(key_cache_ptr + base_kc + idx, vals.to(key_cache_ptr.dtype.element_ty), mask=mask)

    base_v = b * value_ptr.stride(0) + kv_head * value_ptr.stride(1) + s * value_ptr.stride(2)
    vals_v = tl.load(value_ptr + base_v + idx, mask=mask, other=0.0).to(tl.float32)
    base_vc = b * vc_b_stride + kv_head * vc_h_stride + dest * vc_s_stride
    tl.store(value_cache_ptr + base_vc + idx, vals_v.to(value_cache_ptr.dtype.element_ty), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache,
                cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure tensors are on CUDA for Triton
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda \
            and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA device."

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query: sum of squares across head_dim
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               BLOCK_H=128, num_warps=4)

        # 2) Apply RMSNorm to query (bfloat16)
        query_norm = torch.empty_like(query)
        grid_qn = (B * num_q_heads * S,)
        rms_norm_kernel[grid_qn](query, q_norm_weight, query_norm, sum_sums_q,
                                 B, num_q_heads, S, H,
                                 rms_norm_eps,
                                 query.stride(0), query.stride(1), query.stride(2),
                                 query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                 BLOCK_H=128, num_warps=4)

        # 3) Compute cos/sin for rotation per (b, s): shape [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=128, num_warps=4)

        # 4) Apply rotation to RMSNormed query -> rotated_query (bfloat16)
        rotated_query = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, rotated_query,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        rotated_query.stride(0), rotated_query.stride(1), rotated_query.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=128, num_warps=4)

        # 5) Update caches: key_cache[:, :, cache_position] = rotated_query
        #    value_cache[:, :, cache_position] = value (no rotation)
        grid_update = (B * num_kv_heads * S,)
        pos_stride = cache_position  # 1D int64 tensor of length S
        update_cache_kernel[grid_update](rotated_query, value, key_cache, value_cache,
                                         B, num_kv_heads, S,
                                         pos_stride, key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                                         value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                                         BLOCK_H=128, num_warps=4)

        # No return needed; original run updates caches via side effects.
        return rotated_query, rotated_query, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
