import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
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
        offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        total += tl.sum(x32 * x32, axis=0)
    tl.store(sum_sums_ptr + pid, total)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_s = tl.load(sum_sums_ptr + pid).to(tl.float32)
    inv = tl.rsqrt(sum_s / H + 1e-6)  # eps from host
    # Load weight for all H and scale
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w32 = w.to(tl.float32)
        y32 = x32 * inv * w32
        # Store as original dtype (assumed bfloat16 here)
        y = y32.to(x.dtype)
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr,
                              stride_pos0, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * stride_pos0 + s * stride_pos0).to(tl.float32)

    # Compute emb vector of length H: emb[:HALF] = pos * inv_freq, emb[HALF:] = emb[:HALF]
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # First half
        first_mask = idx < HALF
        inv_idx = idx // 2  # integer division
        inv = tl.load(inv_freq_ptr + inv_idx, mask=first_mask, other=0.0).to(tl.float32)
        emb = pos * inv
        # Duplicate for second half
        emb = tl.where(first_mask, emb, emb)
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
    # One program per (b, h, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x
    base_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out

    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)
        cosv = tl.load(cos_ptr + b * cos_stride0 + s * cos_stride1 + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sinv = tl.load(sin_ptr + b * sin_stride0 + s * sin_stride1 + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # rotated = [-x[half:], x[:half]]
        rotated_first = tl.zeros((), dtype=tl.float32)  # placeholder, will fill below
        rotated_second = tl.zeros((), dtype=tl.float32)
        for i in range(0, BLOCK_H):
            j = off + i
            if j < half:
                rotated_first = -tl.load(x_ptr + base_x + (half + j), mask=True, other=0.0).to(tl.float32)
            else:
                rotated_second = tl.load(x_ptr + base_x + (j - half), mask=True, other=0.0).to(tl.float32)

        # Construct rotated vector per lane
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i in range(0, BLOCK_H):
            j = off + i
            if j < half:
                rotated[i] = -tl.load(x_ptr + base_x + (half + j), mask=True, other=0.0).to(tl.float32)
            else:
                rotated[i] = tl.load(x_ptr + base_x + (j - half), mask=True, other=0.0).to(tl.float32)

        y = x * cosv + rotated * sinv
        tl.store(out_ptr + base_out + idx, y.to(tl.float32), mask=mask)  # store in float32 for simplicity; adjust casting as needed


@triton.jit
def update_cache_kernel(x_ptr, out_ptr, dest_ptr, value_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_x, h_stride_x, s_stride_x,
                         batch_stride_out, h_stride_out, s_stride_out,
                         cache_stride0, cache_stride1, cache_stride2,
                         value_stride0, value_stride1, value_stride2,
                         cache_position_ptr,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest_idx = tl.load(cache_position_ptr + s).to(tl.int32)

    base_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x
    base_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out
    base_key = b * cache_stride0 + dest_idx * cache_stride1
    base_val = b * value_stride0 + h * value_stride1 + s * value_stride2

    # 1) Move x (rotated query) into key_cache at dest_idx
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base_key + idx * cache_stride2, x, mask=mask)

    # 2) Move value into value_cache at dest_idx
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        v = tl.load(value_ptr + base_val + idx * value_stride2, mask=mask, other=0.0).to(tl.float32)
        tl.store(dest_ptr + b * cache_stride0 + dest_idx * cache_stride1 + idx * cache_stride2, v, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim, num_attention_heads, num_key_value_heads, max_position_embeddings, cache_len):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.cache_len = cache_len

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Triton-optimized version of the original run function.
        """
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        device = query.device

        # 1) RMSNorm for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid = (B * num_q_heads * S,)
        rms_sum_kernel[grid](query, sum_sums_q,
                             B, num_q_heads, S, H,
                             query.stride(0), query.stride(1), query.stride(2),
                             BLOCK_H=128, num_warps=4)

        query_norm = torch.empty_like(query)
        rms_norm_kernel[grid](query, q_norm_weight, query_norm, sum_sums_q,
                              B, num_q_heads, S, H,
                              query.stride(0), query.stride(1), query.stride(2),
                              query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                              BLOCK_H=128, num_warps=4)

        # 2) Compute rotation sin/cos per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        HALF = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, HALF,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=128, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty_like(query_norm, dtype=torch.float16)  # store in fp16 as per original
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=128, num_warps=4)

        # 4) RMSNorm for key -> key_norm (bfloat16)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](key, sum_sums_k,
                               B, num_kv_heads, S, H,
                               key.stride(0), key.stride(1), key.stride(2),
                               BLOCK_H=128, num_warps=4)

        key_norm = torch.empty_like(key, dtype=torch.float16)
        rms_norm_kernel[grid_k](key, k_norm_weight, key_norm, sum_sums_k,
                                B, num_kv_heads, S, H,
                                key.stride(0), key.stride(1), key.stride(2),
                                key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                BLOCK_H=128, num_warps=4)

        # 5) Apply rotation to key_norm -> key_rot (bfloat16)
        key_rot = torch.empty_like(key_norm, dtype=torch.float16)
        grid_k_rot = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_k_rot](key_norm, cos, sin, key_rot,
                                          B, num_kv_heads, S, H,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=128, num_warps=4)

        # 6) Update caches with rotated keys and values at cache_position
        dest = value_cache  # overwrite values at cache_position
        update_grid = (B * num_kv_heads * S,)
        update_cache_kernel[update_grid](key_rot, query_rot, dest, value,
                                         B, num_kv_heads, S, H,
                                         key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                         query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                         value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                                         value.stride(0), value.stride(1), value.stride(2),
                                         cache_position,
                                         BLOCK_H=128, num_warps=4)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
