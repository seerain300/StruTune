import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B, num_heads, S, H,
                    x_batch_stride, x_head_stride, x_seq_stride,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
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
        offs = b * x_batch_stride + h * x_head_stride + s * x_seq_stride + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals)
    tl.store(sum_sums_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                     B, num_heads, S, H,
                     x_batch_stride, x_head_stride, x_seq_stride,
                     out_batch_stride, out_head_stride, out_seq_stride,
                     weight_stride0,
                     eps, sum_stride0,
                     BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_sums_ptr + pid).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt((sum_val / H) + eps)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * x_batch_stride + h * x_head_stride + s * x_seq_stride + idx
        offs_out = b * out_batch_stride + h * out_head_stride + s * out_seq_stride + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + weight_stride0 + idx, mask=mask, other=1.0).to(tl.float32)
        out_vals = x_vals * w_vals * inv_rms
        tl.store(out_ptr + offs_out, out_vals.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B, S, H,
                              pos_stride0,
                              cos_batch_stride, cos_seq_stride, cos_tail_stride,
                              sin_batch_stride, sin_seq_stride, sin_tail_stride,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    pos = tl.load(position_ids_ptr + b * pos_stride0 + s).to(tl.float32)
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        first_mask = idx < half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate for second half
        emb = tl.where(first_mask, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)
        base = b * cos_batch_stride + s * cos_seq_stride
        tl.store(cos_ptr + base + idx * cos_tail_stride, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_tail_stride, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B, num_heads, S, H,
                           x_batch_stride, x_head_stride, x_seq_stride,
                           out_batch_stride, out_head_stride, out_seq_stride,
                           cos_batch_stride, cos_seq_stride, cos_tail_stride,
                           sin_batch_stride, sin_seq_stride, sin_tail_stride,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base_x = b * x_batch_stride + h * x_head_stride + s * x_seq_stride
    base_out = b * out_batch_stride + h * out_head_stride + s * out_seq_stride
    half = H // 2

    # First half: idx < half, rotated[i] = -x[half + i]
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        x_first = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)
        cos_first = tl.load(cos_ptr + b * cos_batch_stride + s * cos_seq_stride + idx * cos_tail_stride, mask=mask, other=0.0).to(tl.float32)
        sin_first = tl.load(sin_ptr + b * sin_batch_stride + s * sin_seq_stride + idx * sin_tail_stride, mask=mask, other=0.0).to(tl.float32)
        x_rotated_first = -tl.load(x_ptr + base_x + (half + idx), mask=mask, other=0.0).to(tl.float32)
        out_first = x_first * cos_first + x_rotated_first * sin_first
        tl.store(out_ptr + base_out + idx, out_first.to(tl.bfloat16), mask=mask)

    # Second half: idx >= half, rotated[i] = x[idx - half]
    for off in range(0, half, BLOCK_H):
        idx = half + off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_second = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)
        cos_second = tl.load(cos_ptr + b * cos_batch_stride + s * cos_seq_stride + idx * cos_tail_stride, mask=mask, other=0.0).to(tl.float32)
        sin_second = tl.load(sin_ptr + b * sin_batch_stride + s * sin_seq_stride + idx * sin_tail_stride, mask=mask, other=0.0).to(tl.float32)
        x_rotated_second = tl.load(x_ptr + base_x + (idx - half), mask=mask, other=0.0).to(tl.float32)
        out_second = x_second * cos_second + x_rotated_second * sin_second
        tl.store(out_ptr + base_out + idx, out_second.to(tl.bfloat16), mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         cache_pos_ptr,
                         B, num_kv_heads, S, H,
                         query_rot_batch_stride, query_rot_head_stride, query_rot_seq_stride,
                         value_batch_stride, value_head_stride, value_seq_stride,
                         key_cache_batch_stride, key_cache_head_stride, key_cache_seq_stride,
                         value_cache_batch_stride, value_cache_head_stride, value_cache_seq_stride,
                         cache_pos_stride,
                         BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv = (pid % (num_kv_heads * S)) // S
    s = pid % S

    pos = tl.load(cache_pos_ptr + b * cache_pos_stride + s).to(tl.int64)

    base_q = b * query_rot_batch_stride + kv * query_rot_head_stride + s * query_rot_seq_stride
    base_v = b * value_batch_stride + kv * value_head_stride + s * value_seq_stride
    base_k = b * key_cache_batch_stride + kv * key_cache_head_stride + pos * key_cache_seq_stride
    base_v_cache = b * value_cache_batch_stride + kv * value_cache_head_stride + pos * value_cache_seq_stride

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        qr = tl.load(query_rot_ptr + base_q + idx, mask=mask, other=0.0).to(tl.bfloat16)
        v = tl.load(value_ptr + base_v + idx, mask=mask, other=0.0).to(tl.bfloat16)
        # Store into caches
        tl.store(key_cache_ptr + base_k + idx, qr, mask=mask)
        tl.store(value_cache_ptr + base_v_cache + idx, v, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                query: torch.Tensor,
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
        """
        Compute:
          query_norm = RMSNorm(query, q_norm_weight)
          key_norm    = RMSNorm(key, k_norm_weight)
          cos, sin    = rotate embeddings per (b, s) of length H (float32)
          query_rot   = apply rotation to query_norm (bfloat16)
          key_cache[:, :, cache_position] = query_rot
          value_cache[:, :, cache_position] = value
        Returns: query_rot, key_norm (float32), updated key_cache, value_cache
        """
        device = query.device
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query -> query_norm (bfloat16)
        query_norm = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=device)  # we'll produce float32 intermediate and cast at store
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        BLOCK_H = 128
        grid_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               sum_sums_q.stride(0),
                               BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_q](query, q_norm_weight, query_norm, sum_sums_q,
                                B, num_q_heads, S, H,
                                query.stride(0), query.stride(1), query.stride(2),
                                query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                q_norm_weight.stride(0),
                                rms_norm_eps, sum_sums_q.stride(0),
                                BLOCK_H=BLOCK_H, num_warps=4)

        # 2) RMSNorm for key -> key_norm (float32)
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=device)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](key, sum_sums_k,
                               B, num_kv_heads, S, H,
                               key.stride(0), key.stride(1), key.stride(2),
                               sum_sums_k.stride(0),
                               BLOCK_H=BLOCK_H, num_warps=4)

        rms_norm_kernel[grid_k](key, k_norm_weight, key_norm, sum_sums_k,
                                B, num_kv_heads, S, H,
                                key.stride(0), key.stride(1), key.stride(2),
                                key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                k_norm_weight.stride(0),
                                rms_norm_eps, sum_sums_k.stride(0),
                                BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Compute rotation sin/cos per token: [B, S, H], float32
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        rotate_sin_cos_kernel_b_s[(B, S)](position_ids, inv_freq, cos, sin,
                                          B, S, H,
                                          position_ids.stride(0),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 4) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.bfloat16, device=device)
        apply_rotation_kernel[grid_q](query_norm, cos, sin, query_rot,
                                      B, num_q_heads, S, H,
                                      query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                      query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                      cos.stride(0), cos.stride(1), cos.stride(2),
                                      sin.stride(0), sin.stride(1), sin.stride(2),
                                      BLOCK_H=BLOCK_H, num_warps=4)

        # 5) Update caches: key_cache[:, :, cache_position] = query_rot
        #    value_cache[:, :, cache_position] = value
        # Ensure value dtype matches cache dtype (bfloat16)
        value_bf = value.to(torch.bfloat16)
        key_cache_dtype = key_cache.dtype  # typically bfloat16
        value_cache_dtype = value_cache.dtype  # typically bfloat16

        update_cache_kernel[(B * num_kv_heads * S)](
            query_rot, value_bf, key_cache, value_cache, cache_position,
            B, num_kv_heads, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            value_bf.stride(0), value_bf.stride(1), value_bf.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position.stride(0),
            BLOCK_H=BLOCK_H, num_warps=4
        )

        return query_rot, key_norm, key_cache, value_cache


# The following helper functions are not used by ModelNew.forward, but kept for compatibility if needed:

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    cache_len = axes_and_scalars["cache_len"]
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    half_head_dim = 64
    max_position_embeddings = 262144
    rope_theta = 10000000.0
    rms_norm_eps = 1e-6

    query = torch.randn(batch_size, num_attention_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    position_ids = torch.arange(cache_len, cache_len + seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)

    key_cache = torch.randn(batch_size, num_key_value_heads, max_position_embeddings, head_dim, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(batch_size, num_key_value_heads, max_position_embeddings, head_dim, dtype=torch.bfloat16, device=device)

    cache_position = torch.arange(cache_len, cache_len + seq_len, dtype=torch.int64, device=device)

    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))

    return {
        "query": query,
        "key": key,
        "value": value,
        "position_ids": position_ids,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_position": cache_position,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "inv_freq": inv_freq,
        "rms_norm_eps": rms_norm_eps,
    }


@torch.no_grad()
def run(*args):
    # This is a compatibility wrapper; ModelNew.forward is the Triton version.
    return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
