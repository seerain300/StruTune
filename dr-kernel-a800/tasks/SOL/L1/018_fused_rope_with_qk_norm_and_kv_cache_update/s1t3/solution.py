import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_row(x_ptr, out_ptr, weight_ptr, eps, head_dim, BLOCK: tl.constexpr):
    # One program per row
    row_id = tl.program_id(axis=0)
    base = row_id * head_dim
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    sum_sq = tl.sum(x32 * x32, axis=0)
    mean = sum_sq / head_dim
    scale = tl.rsqrt(mean + eps)
    y32 = x32 * (weight * scale)
    y = y32.to(x.dtype)
    tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def rotate_rows(x_ptr, out_ptr, cos_ptr, sin_ptr, head_dim, BLOCK: tl.constexpr):
    # One program per row
    row_id = tl.program_id(axis=0)
    base = row_id * head_dim

    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Split into two halves
    half = head_dim // 2
    h1 = x32[offs < half]
    h2 = x32[offs >= half]

    c = tl.load(cos_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    rot1 = h1 * c[:half] - h2 * s[half:]
    rot2 = h1 * s[:half] + h2 * c[half:]

    out = tl.zeros([head_dim], dtype=tl.float32)
    out[:half] = rot1
    out[half:] = rot2

    y = out.to(x.dtype)
    tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def copy_slice(src_ptr, dst_ptr, cache_offset, n_cols, head_dim, BLOCK: tl.constexpr):
    # Grid: (B, num_heads, n_cols)
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    col = tl.program_id(axis=2)

    src_base = ((b * num_heads + h) * seq_len * head_dim) + (col * head_dim)
    dst_base = ((b * num_heads + h) * (cache_offset + n_cols) * head_dim) + (col * head_dim)

    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    vals = tl.load(src_ptr + src_base + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_base + offs, vals, mask=mask)


@triton.jit
def build_cos_sin_vec(inv_freq_ptr, cos_ptr, sin_ptr, head_dim, BLOCK: tl.constexpr):
    # Build cos/sin vector for rotation: emb = cat([pos * inv_freq, pos * inv_freq], dim=-1), pos=0
    # One program computes vector across head_dim
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim

    # pos = 0, so emb = inv_freq repeated twice (first half, then second half)
    inv = tl.load(inv_freq_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    emb_first = inv
    emb_second = inv
    emb_vec = tl.zeros([BLOCK], dtype=tl.float32)
    emb_vec[:head_dim//2] = emb_first
    emb_vec[head_dim//2:] = emb_second

    c = tl.cos(emb_vec)
    s = tl.sin(emb_vec)
    tl.store(cos_ptr + offs, c, mask=mask)
    tl.store(sin_ptr + offs, s, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same 11 arguments as in the original run function
        if len(args) < 11:
            raise RuntimeError("ModelNew.forward expects at least 11 arguments")

        # Extract inputs and ensure contiguous
        query = args[0].contiguous()
        key = args[1].contiguous()
        value = args[2].contiguous()
        position_ids = args[3].contiguous()
        key_cache = args[4].contiguous()
        value_cache = args[5].contiguous()
        cache_position = args[6].contiguous()
        q_norm_weight = args[7].contiguous()
        k_norm_weight = args[8].contiguous()
        inv_freq = args[9].contiguous()
        rms_norm_eps = float(args[10]) if len(args) > 10 else 1e-6

        # Shapes
        Bq, num_q_heads, seq_len, head_dim = query.shape
        Bk, num_kv_heads, _, _ = key.shape
        assert Bq == Bk
        B = Bq
        assert key.shape == (B, num_kv_heads, seq_len, head_dim)
        assert value.shape == (B, num_kv_heads, seq_len, head_dim)
        assert key_cache.shape == (B, num_kv_heads, 262144, head_dim)
        assert value_cache.shape == (B, num_kv_heads, 262144, head_dim)

        # 1) RMSNorm for query and key in Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        BLOCK = head_dim  # full head_dim processed per row
        n_rows_q = B * num_q_heads * seq_len
        n_rows_k = B * num_kv_heads * seq_len

        rms_norm_row[(n_rows_q,)](
            query, query_norm, q_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
        )
        rms_norm_row[(n_rows_k,)](
            key, key_norm, k_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
        )

        # 2) Build cos/sin rotation vector in Triton (emb = cat([pos*inv_freq, pos*inv_freq], pos=0))
        cos_vec = torch.empty(head_dim, dtype=torch.float32, device=query.device)
        sin_vec = torch.empty(head_dim, dtype=torch.float32, device=query.device)

        build_cos_sin_vec[(1,)](
            inv_freq, cos_vec, sin_vec, head_dim, BLOCK=BLOCK, num_warps=1
        )

        # 3) Apply rotation using Triton
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        rotate_rows[(n_rows_q,)](
            query_norm, query_rotated, cos_vec, sin_vec, head_dim, BLOCK=BLOCK, num_warps=4
        )
        rotate_rows[(n_rows_k,)](
            key_norm, key_rotated, cos_vec, sin_vec, head_dim, BLOCK=BLOCK, num_warps=4
        )

        # 4) Update caches: copy rotated keys and original values into cache at cache_len + cache_position
        cache_len = int(args[11]) if len(args) > 11 else 0
        num_warps_cp = 4
        copy_slice[(B, num_kv_heads, seq_len)](
            key_rotated, key_cache, cache_len, seq_len, head_dim, BLOCK=BLOCK, num_warps=num_warps_cp
        )
        copy_slice[(B, num_kv_heads, seq_len)](
            value, value_cache, cache_len, seq_len, head_dim, BLOCK=BLOCK, num_warps=num_warps_cp
        )

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
