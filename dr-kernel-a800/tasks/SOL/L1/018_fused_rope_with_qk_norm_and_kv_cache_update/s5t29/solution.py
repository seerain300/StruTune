import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_rope_update(
    query, key, value,
    query_out, key_out, value_out,
    q_norm_weight, k_norm_weight,
    inv_freq,  # [HALF] tensor, dtype float32
    key_cache, value_cache,
    B, S,
    num_q_heads, num_kv_heads,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map pid to (b, h, s) for query/key
    tmp = pid
    s = tmp % S
    tmp = tmp // S
    h = tmp % num_q_heads
    b = tmp // num_q_heads

    base_q = b * D * num_q_heads + h * D
    base_k = b * D * num_q_heads + h * D

    # RMSNorm for query
    sumsq_q = 0.0
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(query + base_q + offs, mask=mask, other=0.0)
        xv = x.to(tl.float32)
        sumsq_q += tl.sum(xv * xv, axis=0)
    mean_q = sumsq_q / D
    scale_q = 1.0 / tl.sqrt(mean_q + 0.0)
    wq = tl.load(q_norm_weight + tl.arange(0, D), mask=(tl.arange(0, D) < D), other=0.0).to(tl.float32)

    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(query + base_q + offs, mask=mask, other=0.0)
        xv = x.to(tl.float32)
        y = (xv * scale_q) * wq
        tl.store(query_out + base_q + offs, y.to(x.dtype), mask=mask)

    # RotE for query
    pos = cache_len + s  # S is used as cache_len
    idx = tl.arange(0, HALF)
    base = (pos * tl.load(inv_freq + idx)).to(tl.float32)
    double = base * 2.0
    emb = tl.concatenate([base, double])  # [D]
    cos = tl.cos(emb)
    sin = tl.sin(emb)

    first_half_cos = cos[:HALF]
    first_half_sin = sin[:HALF]
    second_half_cos = cos[HALF:]
    second_half_sin = sin[HALF:]

    # First 64 dims: y1 = xv[:64] * cos[:64] + (-xv[64:]) * sin[:64]
    for d in range(0, HALF, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < HALF
        xv1 = tl.load(query_out + base_q + offs, mask=mask, other=0.0).to(tl.float32)
        xv2 = tl.load(query_out + base_q + (offs + HALF), mask=mask, other=0.0).to(tl.float32)
        c = first_half_cos[offs]
        s = first_half_sin[offs]
        y1 = xv1 * c + (-xv2) * s
        tl.store(query_out + base_q + offs, y1.to(query_out.dtype), mask=mask)

    # Second 64 dims: y2 = xv[:64] * cos[64:] + (-xv[64:]) * sin[48:64]
    for d in range(0, HALF, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < HALF
        xv1 = tl.load(query_out + base_q + offs, mask=mask, other=0.0).to(tl.float32)
        xv2 = tl.load(query_out + base_q + (offs + HALF), mask=mask, other=0.0).to(tl.float32)
        c = second_half_cos[offs]
        s = second_half_sin[offs]
        y2 = xv1 * c + (-xv2) * s
        tl.store(query_out + base_q + (offs + HALF), y2.to(query_out.dtype), mask=mask)

    # RMSNorm for key: similar
    sumsq_k = 0.0
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        xk = tl.load(key + base_k + offs, mask=mask, other=0.0)
        xkv = xk.to(tl.float32)
        sumsq_k += tl.sum(xkv * xkv, axis=0)
    mean_k = sumsq_k / D
    scale_k = 1.0 / tl.sqrt(mean_k + 0.0)
    wk = tl.load(k_norm_weight + tl.arange(0, D), mask=(tl.arange(0, D) < D), other=0.0).to(tl.float32)

    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        xk = tl.load(key + base_k + offs, mask=mask, other=0.0)
        xkv = xk.to(tl.float32)
        yk = (xkv * scale_k) * wk
        tl.store(key_out + base_k + offs, yk.to(key_out.dtype), mask=mask)

    # RotE for key: same emb, cos, sin
    first_half_cos = cos[:HALF]
    first_half_sin = sin[:HALF]
    second_half_cos = cos[HALF:]
    second_half_sin = sin[HALF:]

    for d in range(0, HALF, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < HALF
        xk1 = tl.load(key_out + base_k + offs, mask=mask, other=0.0).to(tl.float32)
        xk2 = tl.load(key_out + base_k + (offs + HALF), mask=mask, other=0.0).to(tl.float32)
        c = first_half_cos[offs]
        s = first_half_sin[offs]
        y1k = xk1 * c + (-xk2) * s
        tl.store(key_out + base_k + offs, y1k.to(key_out.dtype), mask=mask)
    for d in range(0, HALF, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < HALF
        xk1 = tl.load(key_out + base_k + offs, mask=mask, other=0.0).to(tl.float32)
        xk2 = tl.load(key_out + base_k + (offs + HALF), mask=mask, other=0.0).to(tl.float32)
        c = second_half_cos[offs]
        s = second_half_sin[offs]
        y2k = xk1 * c + (-xk2) * s
        tl.store(key_out + base_k + (offs + HALF), y2k.to(key_out.dtype), mask=mask)

    # Update caches at position (b, h, cache_len + s)
    base_kv = b * D * num_kv_heads + h * D
    pos_idx = cache_len + s
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        xk_out = tl.load(key_out + base_k + offs, mask=mask, other=0.0)
        xk_out_f32 = xk_out.to(tl.float32)
        tl.store(key_cache + base_kv + pos_idx * D + offs, xk_out_f32.to(key_cache.dtype), mask=mask)

    base_v = b * D * num_kv_heads + h * D
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        xv = tl.load(value + base_v + s * D + offs, mask=mask, other=0.0)
        xv_f32 = xv.to(tl.float32)
        tl.store(value_cache + base_v + pos_idx * D + offs, xv_f32.to(value_cache.dtype), mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        query = args[0].contiguous()
        key = args[1].contiguous()
        value = args[2].contiguous()
        q_norm_weight = args[7].contiguous()
        k_norm_weight = args[8].contiguous()
        inv_freq = args[9].contiguous()
        key_cache = args[4].contiguous()
        value_cache = args[5].contiguous()
        # cache_len is args[6], but not used in kernel (we use S as cache_len)
        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        HALF = D // 2
        query_out = torch.empty_like(query)
        key_out = torch.empty_like(query)
        value_out = torch.empty_like(value)
        grid = (B * num_q_heads * S,)
        rmsnorm_rope_update[grid](
            query, key, value,
            query_out, key_out, value_out,
            q_norm_weight, k_norm_weight,
            inv_freq,
            key_cache, value_cache,
            B, S,
            num_q_heads, 8,
            D=D, HALF=HALF, BLOCK_D=D,
            num_warps=4, num_stages=2,
        )
        return query_out, key_out, value_out, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
