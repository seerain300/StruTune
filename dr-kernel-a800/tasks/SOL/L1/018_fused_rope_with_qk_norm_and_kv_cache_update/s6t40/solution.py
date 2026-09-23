import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, seq_stride_x):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = 0.0
    for off in range(0, H, 128):
        idx = off + tl.arange(0, 128)
        mask = idx < H
        offs = b * batch_stride_x + h * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, inv_rms_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, head_stride_x, seq_stride_x,
                    batch_stride_out, head_stride_out, seq_stride_out,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    inv_rms = tl.load(inv_rms_ptr + pid)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x_vals * inv_rms
        y = y * w_vals
        offs_out = b * batch_stride_out + h * head_stride_out + s * seq_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(pos_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ids_ptr + b * pos_ids_ptr.stride(0) + s * pos_ids_ptr.stride(1)).to(tl.float32)

    # emb: first half
    for i in range(0, H2):
        v = pos * tl.load(inv_freq_ptr + i).to(tl.float32)
        tl.store(cos_ptr + b * cos_ptr.stride(0) + s * cos_ptr.stride(1) + i * cos_ptr.stride(2), v)
        tl.store(sin_ptr + b * sin_ptr.stride(0) + s * sin_ptr.stride(1) + i * sin_ptr.stride(2), 0.0)
    # emb: second half duplicates first half
    for i in range(0, H2):
        v = tl.load(cos_ptr + b * cos_ptr.stride(0) + s * cos_ptr.stride(1) + i * cos_ptr.stride(2))
        tl.store(cos_ptr + b * cos_ptr.stride(0) + s * cos_ptr.stride(1) + (i + H2) * cos_ptr.stride(2), v)
        tl.store(sin_ptr + b * sin_ptr.stride(0) + s * sin_ptr.stride(1) + (i + H2) * sin_ptr.stride(2), 0.0)
    # Note: We compute cos and sin in host since Triton lacks sin/cos; this kernel initializes the structure. We'll set sin to zeros and compute in host separately.
    return


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, head_stride_x, seq_stride_x,
                           batch_stride_out, head_stride_out, seq_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
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
        offs_x = b * batch_stride_x + h * head_stride_x + s * seq_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in range(0, BLOCK_H):
            if i < half:
                rotated[i] = -tl.load(x_ptr + offs_x + (half + i), mask=True, other=0.0).to(tl.float32)
            else:
                rotated[i] = tl.load(x_ptr + offs_x + (i - half), mask=True, other=0.0).to(tl.float32)

        y = x_vals * cos_vals + rotated * sin_vals

        offs_out = b * batch_stride_out + h * head_stride_out + s * seq_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def update_cache_kernel(x_ptr, cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_x, head_stride_x, seq_stride_x,
                         batch_stride_cache, head_stride_cache, seq_stride_cache,
                         BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    dest = tl.load(cache_pos_ptr + s).to(tl.int32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * head_stride_x + s * seq_stride_x + idx
        vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        offs_cache = b * batch_stride_cache + h * head_stride_cache + dest * seq_stride_cache + idx
        tl.store(cache_ptr + offs_cache, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position,
        #       q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        assert len(args) == 11, "Expected 11 inputs"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args
        device = query.device
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query (bfloat16 output)
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=128, num_warps=4
        )
        inv_rms_q = 1.0 / torch.sqrt((sum_sums_q / H) + rms_norm_eps)
        query_norm = torch.empty_like(query)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm, inv_rms_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 2) Compute cos/sin on host for rotation (float32) and pass to Triton
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        H2 = H // 2
        for b in range(B):
            for s in range(S):
                pos = int(position_ids[b, s].item())
                angles = torch.empty(H, device=device, dtype=torch.float32)
                for i in range(H2):
                    angles[i] = pos * float(inv_freq[i].item())
                for i in range(H2):
                    angles[i + H2] = angles[i]
                c = torch.cos(angles)
                s_ = torch.sin(angles)
                cos[b, s, :] = c
                sin[b, s, :] = s_

        # 3) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # 4) RMSNorm for key (bfloat16 output)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_sums_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=128, num_warps=4
        )
        inv_rms_k = 1.0 / torch.sqrt((sum_sums_k / H) + rms_norm_eps)
        key_norm = torch.empty_like(key)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm, inv_rms_k,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            BLOCK_H=128, num_warps=4
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
            BLOCK_H=128, num_warps=4
        )

        # 6) Update caches: key_cache[:, :, cache_position] = key_rot; value_cache[:, :, cache_position] = value
        update_cache_kernel[(B * num_kv_heads * S,)](
            key_rot, key_cache, cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            BLOCK_H=128, num_warps=4
        )

        # Update value cache: value_cache[:, :, cache_position] = value
        update_cache_kernel[(B * num_kv_heads * S,)](
            value, value_cache, cache_position,
            B, num_kv_heads, S, H,
            value.stride(0), value.stride(1), value.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            BLOCK_H=128, num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
