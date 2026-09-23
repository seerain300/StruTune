import torch
import triton
import triton.language as tl


@triton.jit
def emb_and_apply(
    query_in_ptr, query_out_ptr,  # for query processing
    key_in_ptr, key_out_ptr,      # for key processing
    value_in_ptr, value_out_ptr,  # for value processing
    key_cache_ptr, value_cache_ptr,  # cache pointers (written only when processing key/value)
    q_norm_w_ptr, k_norm_w_ptr,    # RMSNorm weights
    inv_freq_ptr,                  # [HALF] float32
    B: tl.constexpr,               # batch size
    S: tl.constexpr,               # seq_len
    D: tl.constexpr,               # head_dim (128)
    num_q_heads: tl.constexpr,     # number of query heads
    num_kv_heads: tl.constexpr,    # number of kv heads
    cache_len: tl.constexpr,       # cache length
    HALF: tl.constexpr,            # D//2 (64)
    TOTAL_PROGS: tl.constexpr,     # total program instances (B * (num_q_heads + num_kv_heads) * S)
    BLOCK_D: tl.constexpr = 128    # block for D
):
    # Each program handles one (b, head, s). We decode pid to determine if it's query or kv.
    pid = tl.program_id(0)
    # Determine if this pid is in the query range
    is_query = pid < (B * num_q_heads * S)
    # Compute b, head, s for the selected range
    if is_query:
        # Decode pid for query: pid in [0, B*num_q_heads*S)
        b = pid // (num_q_heads * S)
        rem = pid % (num_q_heads * S)
        h = rem // S
        s = rem % S

        # 1) RMSNorm for query
        # x: [B, num_q_heads, S, D]
        x = tl.load(query_in_ptr + b * (num_q_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D))
        x32 = x.to(tl.float32)
        sumsq = tl.sum(x32 * x32)
        mean = sumsq / D
        scale = 1.0 / tl.sqrt(mean + 1e-6)
        w = tl.load(q_norm_w_ptr + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D, other=1.0).to(tl.float32)
        y = x32 * scale * w
        tl.store(query_out_ptr + b * (num_q_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D), y.to(x.dtype))

        # 2) RotE for query
        pos_val = cache_len + s
        pos_f = pos_val.to(tl.float32)
        inv = tl.load(inv_freq_ptr + tl.arange(0, HALF))
        # emb = [pos*inv, pos*inv]
        emb = tl.zeros([D], dtype=tl.float32)
        emb[:HALF] = pos_f * inv
        emb[HALF:] = pos_f * inv
        cos = tl.cos(emb)
        sin = tl.sin(emb)

        x_norm = tl.load(query_out_ptr + b * (num_q_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D))  # normalized query
        x_norm32 = x_norm.to(tl.float32)
        x1 = x_norm32[:HALF]
        x2 = x_norm32[HALF:]
        c1 = cos[:HALF]
        s1 = sin[:HALF]
        c2 = cos[HALF:]
        s2 = sin[HALF:]

        out1 = x1 * c1 + (-x2) * s1
        out2 = x2 * c2 + (x1) * s2
        out32 = tl.zeros([D], dtype=tl.float32)
        out32[:HALF] = out1
        out32[HALF:] = out2
        out = out32.to(x_norm.dtype)
        tl.store(query_out_ptr + b * (num_q_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D), out)

    else:
        # Decode pid for key/value: pid in [B*num_q_heads*S, TOTAL_PROGS)
        kv_base = B * num_q_heads * S
        b = (pid - kv_base) // (num_kv_heads * S)
        rem = (pid - kv_base) % (num_kv_heads * S)
        h = rem // S
        s = rem % S

        # a) RMSNorm for key
        k = tl.load(key_in_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D))
        k32 = k.to(tl.float32)
        sumsq = tl.sum(k32 * k32)
        mean = sumsq / D
        scale = 1.0 / tl.sqrt(mean + 1e-6)
        w = tl.load(k_norm_w_ptr + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D, other=1.0).to(tl.float32)
        y = k32 * scale * w
        tl.store(key_out_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D), y.to(k.dtype))

        # b) RotE for key (value is stored unchanged in cache)
        pos_val = cache_len + s
        pos_f = pos_val.to(tl.float32)
        inv = tl.load(inv_freq_ptr + tl.arange(0, HALF))
        emb = tl.zeros([D], dtype=tl.float32)
        emb[:HALF] = pos_f * inv
        emb[HALF:] = pos_f * inv
        cos = tl.cos(emb)
        sin = tl.sin(emb)

        k_norm = tl.load(key_out_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D))
        k_norm32 = k_norm.to(tl.float32)
        x1 = k_norm32[:HALF]
        x2 = k_norm32[HALF:]
        c1 = cos[:HALF]
        s1 = sin[:HALF]
        c2 = cos[HALF:]
        s2 = sin[HALF:]
        out1 = x1 * c1 + (-x2) * s1
        out2 = x2 * c2 + (x1) * s2
        out32 = tl.zeros([D], dtype=tl.float32)
        out32[:HALF] = out1
        out32[HALF:] = out2
        out = out32.to(k_norm.dtype)
        tl.store(key_out_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D), out)

        # c) Update caches: rotated key into key_cache, original value into value_cache
        # key_cache layout: [B, num_kv_heads, max_pos, D], we store at row (b, h, cache_len + s, :)
        v = tl.load(value_in_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D))
        tl.store(value_cache_ptr + b * (num_kv_heads * D * 262144) + h * (262144 * D) + (cache_len + s) * D + tl.arange(0, BLOCK_D), v.to(v.dtype))
        # Note: value_cache_ptr stride uses max_pos * D; Triton infers element size, we store a vector of length D.
        # However, for simplicity and to avoid pointer arithmetic issues with max_pos, we can compute exact address:
        # value_cache_ptr is a flat contiguous pointer; we need to pass the correct stride. Triton expects pointer to memory, so we compute pointer as:
        # For each (b, h), we have base = b * (num_kv_heads * max_pos * D) + h * (max_pos * D), then offset (cache_len + s) * D.
        base_val = b * (num_kv_heads * D * 262144) + h * (262144 * D)
        offset_val = (cache_len + s) * D
        tl.store(value_cache_ptr + base_val + offset_val + tl.arange(0, BLOCK_D), v.to(v.dtype))

        base_key = b * (num_kv_heads * D * 262144) + h * (262144 * D)
        offset_key = (cache_len + s) * D
        tl.store(key_cache_ptr + base_key + offset_key + tl.arange(0, BLOCK_D), tl.load(key_out_ptr + b * (num_kv_heads * S * D) + h * (S * D) + s * D + tl.arange(0, BLOCK_D)))


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0]
        key = args[1]
        value = args[2]
        key_cache = args[4]
        value_cache = args[5]
        q_norm_weight = args[7]
        k_norm_weight = args[8]
        inv_freq = args[9]

        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]

        num_kv_heads = key.shape[1]
        HALF = D // 2

        # Allocate outputs
        query_out = torch.empty_like(query)
        key_out = torch.empty_like(key)
        value_out = torch.empty_like(value)

        # Launch kernel for query: pid in [0, B*num_q_heads*S)
        grid_query = (B * num_q_heads * S,)
        emb_and_apply[grid_query](
            query, query_out,
            key, key_out,
            value, value_out,
            key_cache, value_cache,
            q_norm_weight, k_norm_weight,
            inv_freq,
            B, S, D, num_q_heads, num_kv_heads, 0, B * (num_q_heads + num_kv_heads) * S, 128
        )

        # Launch kernel for key/value:


def run(*args):
    return ModelNew()(*args)
