import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     eps: tl.constexpr, BLOCK_H: tl.constexpr):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_ptr + pid)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H, other=1.0).to(tl.float32) * inv_rms

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_in = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx

        x_vals = tl.load(x_ptr + offs_in, mask=mask, other=0.0).to(tl.float32)
        y_vals = x_vals * scale
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rms_sum_key_kernel(k_ptr, sum_k_ptr,
                       B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                       batch_stride_k, h_stride_k, s_stride_k,
                       BLOCK_H: tl.constexpr):
    # One program per (b, kv_h, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride_k + kv_h * h_stride_k + s * s_stride_k + idx
        k_vals = tl.load(k_ptr + offs, mask=mask, other=0.0)
        k_vals = k_vals.to(tl.float32)
        sum_sq += tl.sum(k_vals * k_vals, axis=0)
    tl.store(sum_k_ptr + pid, sum_sq)


@triton.jit
def rms_norm_key_kernel(k_ptr, k_weight_ptr, out_k_ptr, sum_k_ptr,
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_k, h_stride_k, s_stride_k,
                        batch_stride_out_k, h_stride_out_k, s_stride_out_k,
                        eps: tl.constexpr, BLOCK_H: tl.constexpr):
    # One program per (b, kv_h, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_k_ptr + pid)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(k_weight_ptr + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H, other=1.0).to(tl.float32) * inv_rms

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_in = b * batch_stride_k + kv_h * h_stride_k + s * s_stride_k + idx
        offs_out = b * batch_stride_out_k + kv_h * h_stride_out_k + s * s_stride_out_k + idx

        k_vals = tl.load(k_ptr + offs_in, mask=mask, other=0.0).to(tl.float32)
        y_vals = k_vals * scale
        tl.store(out_k_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                              pos_stride,  # stride for position_ids [B, S]
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # Each program handles one (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_half = idx < H2
        # inv_freq for first half only
        inv_freq_idx = tl.where(first_half, idx // 2, tl.zeros((), dtype=tl.int32))
        inv_freq_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate second half
        emb = tl.where(first_half, emb_first, emb_first)

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

    base = b * cos_stride0 + s * cos_stride1
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # Load x
        offs_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        first_mask = idx < half
        second_mask = idx >= half
        x_first = tl.load(x_ptr + offs_x, mask=first_mask, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + offs_x, mask=second_mask, other=0.0).to(tl.float32)

        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        rotated = tl.where(first_mask, x_first, -x_second)

        y_vals = x_vals * cos_vals + rotated * sin_vals

        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(x_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        dest_ptr,  # cache_position [S]
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_x, h_stride_x, s_stride_x, dest_stride,
                        batch_stride_value, h_stride_value, s_stride_value, dest_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride_kc, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        BLOCK_H: tl.constexpr):
    # One program per (b, kv_h, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv_h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest = tl.load(dest_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        src_offs_x = b * batch_stride_x + kv_h * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_rot_ptr + src_offs_x, mask=mask, other=0.0).to(tl.float32)
        tl.store(key_cache_ptr + b * batch_stride_kc + kv_h * h_stride_kc + dest * dest_stride_kc + idx * d_stride_kc,
                 x_vals, mask=mask)

        src_offs_v = b * batch_stride_value + kv_h * h_stride_value + s * s_stride_value + idx
        v_vals = tl.load(value_ptr + src_offs_v, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + b * batch_stride_vc + kv_h * h_stride_vc + dest * dest_stride_vc + idx * d_stride_vc,
                 v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value,
                position_ids, key_cache, value_cache,
                cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]
        H2 = H // 2
        device = query.device
        dtype = query.dtype

        # 1) RMSNorm for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        grid_sum_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_sum_q](query, sum_sums_q,
                                   B, num_q_heads, S, H,
                                   query.stride(0), query.stride(1), query.stride(2),
                                   BLOCK_H=64, num_warps=4)
        query_norm = torch.empty_like(query)
        grid_norm_q = (B * num_q_heads * S, H)
        rms_norm_kernel[grid_norm_q](query, q_norm_weight, query_norm, sum_sums_q,
                                     B, num_q_heads, S, H,
                                     query.stride(0), query.stride(1), query.stride(2),
                                     query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                     rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 2) Compute cos/sin per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, H2,
                                           position_ids.stride(0),  # pos_stride
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=64, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot
        query_rot = torch.empty_like(query_norm)
        grid_rot = (B * num_q_heads * S, H)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=64, num_warps=4)

        # 4) RMSNorm for key
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_key_kernel[grid_sum_k](key, sum_sums_k,
                                       B, num_kv_heads, S, H,
                                       key.stride(0), key.stride(1), key.stride(2),
                                       BLOCK_H=64, num_warps=4)
        key_norm = torch.empty_like(key)
        grid_norm_k = (B * num_kv_heads * S, H)
        rms_norm_key_kernel[grid_norm_k](key, k_norm_weight, key_norm, sum_sums_k,
                                         B, num_kv_heads, S, H,
                                         key.stride(0), key.stride(1), key.stride(2),
                                         key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                         rms_norm_eps, BLOCK_H=64, num_warps=4)

        # 5) Update caches: key_cache[:, :, cache_position] = query_rot; value_cache[:, :, cache_position] = value
        dest = cache_position  # [S], int64
        grid_upd = (B * num_kv_heads * S,)
        update_cache_kernel[grid_upd](query_rot, value,
                                      key_cache, value_cache,
                                      dest,
                                      B, num_kv_heads, S, H,
                                      query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), dest.stride(0),
                                      value.stride(0), value.stride(1), value.stride(2), dest.stride(0),
                                      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                                      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                                      BLOCK_H=64, num_warps=4)

        return query_rot, key_norm, key_cache, value_cache


# For compatibility with the provided get_inputs utility:
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    cache_len = axes_and_scalars["cache_len"]
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    half_head_dim = head_dim // 2
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

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_head_dim, dtype=torch.float32, device=device) / half_head_dim))

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


# Example usage (not required by evaluator, but useful for testing):
# model = ModelNew().cuda()
# inputs = get_inputs({"batch_size": 1, "seq_len": 256, "cache_len": 128}, torch.device("cuda"))
# query_rot, key_norm, key_cache, value_cache = model(
#     inputs["query"], inputs["key"], inputs["value"],
#     inputs["position_ids"], inputs["key_cache"], inputs["value_cache"],
#     inputs["cache_position"], inputs["q_norm_weight"], inputs["k_norm_weight"], inputs["inv_freq"], inputs["rms_norm_eps"]
# )


def run(*args):
    return ModelNew()(*args)
