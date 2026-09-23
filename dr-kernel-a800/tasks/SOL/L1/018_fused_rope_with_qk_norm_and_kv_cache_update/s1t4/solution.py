import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_row(x_ptr, y_ptr, w_ptr, eps, head_dim: tl.constexpr, BLOCK: tl.constexpr, num_warps: tl.constexpr):
    """
    RMSNorm per row: y = w * x / sqrt(mean(x^2) + eps)
    Assumes x is row-major flattened with BLOCK=HEAD_DIM per row.
    One Triton program handles one row (program_id(0) = row index).
    """
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x_row = tl.load(x_ptr + row_id * BLOCK + idx)
    x_fp32 = x_row.to(tl.float32)
    var = tl.sum(x_fp32 * x_fp32, axis=0) / BLOCK
    scale = 1.0 / tl.sqrt(var + eps)
    w_row = tl.load(w_ptr + idx).to(tl.float32)
    y_row = (x_fp32 * w_row) * scale
    y_row = y_row.to(x_row.dtype)
    tl.store(y_ptr + row_id * BLOCK + idx, y_row)


@triton.jit
def build_cos_sin_per_row(x_ptr, cos_ptr, sin_ptr, inv_freq_ptr, head_dim: tl.constexpr, BLOCK: tl.constexpr, num_warps: tl.constexpr):
    """
    Build cos and sin vectors for a single row using its position (from x_ptr), inv_freq_ptr, and output cos_ptr, sin_ptr.
    x_ptr[row] holds the position (int64). We compute emb = pos * inv_freq[:head_dim//2],
    then emb2 = cat([emb, emb], -1) -> length head_dim, and cos/sin = cos(emb2), sin(emb2).
    """
    row_id = tl.program_id(0)
    # Load position as int64
    pos = tl.load(x_ptr + row_id).to(tl.float32)
    ar = tl.arange(0, head_dim // 2)
    inv = tl.load(inv_freq_ptr + ar).to(tl.float32)
    emb = pos * inv  # length head_dim//2
    # Duplicate: emb2 = [emb, emb]
    idx = tl.arange(0, head_dim)
    idx_half = idx < (head_dim // 2)
    emb2 = tl.where(idx_half, emb, emb)
    cos_vec = tl.cos(emb2)
    sin_vec = tl.sin(emb2)
    tl.store(cos_ptr + idx, cos_vec)
    tl.store(sin_ptr + idx, sin_vec)


@triton.jit
def rotate_rows(x_ptr, y_ptr, cos_ptr, sin_ptr, head_dim: tl.constexpr, BLOCK: tl.constexpr, num_warps: tl.constexpr):
    """
    Apply rotation to rows: for each row,
      c = cos[:], s = sin[:], x = x_ptr[row], y = y_ptr[row]
      x1 = x[:head_dim//2], x2 = x[head_dim//2:], xh = [-x2, x1]
      y = x * c - xh * s
    Assumes y and x have the same shape and head_dim.
    """
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row_id * BLOCK + idx)
    c = tl.load(cos_ptr + idx).to(tl.float32)
    s = tl.load(sin_ptr + idx).to(tl.float32)
    half = head_dim // 2
    x1 = x[:half]
    x2 = x[half:]
    xh = tl.cat([-x2, x1], axis=0)  # length head_dim
    y = x.to(tl.float32) * c - xh.to(tl.float32) * s
    y = y.to(x.dtype)
    tl.store(y_ptr + row_id * BLOCK + idx, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-only forward replacing the original run.
        Expected args: query, key, value, position_ids, key_cache, value_cache,
                       cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        Returns: (query_rotated, key_rotated, key_cache, value_cache)
        """
        if len(args) < 11:
            raise RuntimeError("ModelNew.forward expects at least 11 arguments")

        # Ensure all tensors are on the same device and contiguous
        query = args[0].contiguous()
        key = args[1].contiguous()
        value = args[2].contiguous()
        position_ids = args[3].contiguous()  # [B, seq_len], int64
        key_cache = args[4].contiguous()     # [B, num_kv_heads, 262144, head_dim], bfloat16
        value_cache = args[5].contiguous()   # [B, num_kv_heads, 262144, head_dim], bfloat16
        cache_position = args[6].contiguous()  # [seq_len], int64
        q_norm_weight = args[7].contiguous()   # [head_dim], bfloat16
        k_norm_weight = args[8].contiguous()   # [head_dim], bfloat16
        inv_freq = args[9].contiguous()        # [head_dim//2], float32
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

        # 1) RMSNorm for query and key (Triton), guard for empty rows
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        BLOCK = head_dim  # full head_dim processed per row
        n_rows_q = B * num_q_heads * seq_len
        n_rows_k = B * num_kv_heads * seq_len

        if n_rows_q > 0:
            rms_norm_row[(n_rows_q,)](
                query, query_norm, q_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
            )
        if n_rows_k > 0:
            rms_norm_row[(n_rows_k,)](
                key, key_norm, k_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
            )

        # 2) Build cos/sin per row in Triton: use position_ids to compute rotation (Triton-only).
        # Create position buffers for each row (int64)
        pos_query = torch.empty(n_rows_q, dtype=torch.int64, device=query.device)
        pos_key = torch.empty(n_rows_k, dtype=torch.int64, device=key.device)

        # Fill pos buffers: pos[row] = position_ids[batch, token]
        # For query
        if n_rows_q > 0:
            for row in range(n_rows_q):
                batch = row // (num_q_heads * seq_len)
                token = row % seq_len
                pos_query[row] = position_ids[batch, token].item()
        # For key
        if n_rows_k > 0:
            for row in range(n_rows_k):
                batch = row // (num_kv_heads * seq_len)
                token = row % seq_len
                pos_key[row] = position_ids[batch, token].item()

        # Allocate cos/sin vectors per row (fp32)
        cos_query = torch.empty(n_rows_q * head_dim, dtype=torch.float32, device=query.device)
        sin_query = torch.empty(n_rows_q * head_dim, dtype=torch.float32, device=query.device)
        cos_key = torch.empty(n_rows_k * head_dim, dtype=torch.float32, device=key.device)
        sin_key = torch.empty(n_rows_k * head_dim, dtype=torch.float32, device=key.device)

        # Build cos/sin per row in Triton
        if n_rows_q > 0:
            build_cos_sin_per_row[(n_rows_q,)](
                pos_query, cos_query, sin_query, inv_freq, head_dim, BLOCK=head_dim, num_warps=1
            )
        if n_rows_k > 0:
            build_cos_sin_per_row[(n_rows_k,)](
                pos_key, cos_key, sin_key, inv_freq, head_dim, BLOCK=head_dim, num_warps=1
            )

        # 3) Apply rotation using Triton
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        if n_rows_q > 0:
            rotate_rows[(n_rows_q,)](
                query_norm, query_rotated, cos_query, sin_query, head_dim, BLOCK=BLOCK, num_warps=4
            )
        if n_rows_k > 0:
            rotate_rows[(n_rows_k,)](
                key_norm, key_rotated, cos_key, sin_key, head_dim, BLOCK=BLOCK, num_warps=4
            )

        # 4) Update caches using PyTorch advanced indexing (Triton cannot index by cache_position)
        cache_len = int(args[6].max().item()) if args[6].numel() > 0 else 0  # not used in indexing below

        # Ensure cache_position is int32 for computing indices
        cache_position = cache_position.to(torch.int32)

        # Copy rotated keys into key_cache at positions cache_len + cache_position[t]
        for b in range(B):
            for h in range(num_kv_heads):
                # Rotated keys per (batch, head): shape [seq_len, head_dim]
                keys_b_h = key_rotated[b, h]  # [seq_len, head_dim]
                t = torch.arange(seq_len, device=keys_b_h.device)
                p = cache_len + cache_position[t]  # [seq_len], int32 positions in cache
                key_cache[b, h, p, :] = keys_b_h  # advanced indexing: place entire rows

                # Copy original values into value_cache at the same positions
                vals_b_h = value[b, h]  # [seq_len, head_dim]
                value_cache[b, h, p, :] = vals_b_h

        # Return rotated tensors and updated caches
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
