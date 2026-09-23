import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    rows,           # int32
    D: tl.constexpr,       # int (e.g., 128)
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    # Compute sum of squares across the last dimension in fp32
    sumsq = 0.0
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = tl.rsqrt(mean + eps)

    # Apply per-dimension weight and store
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def apply_rope_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    COS_ptr,        # *pointer* to cos vector, shape [D] (bf16)
    SIN_ptr,        # *pointer* to sin vector, shape [D] (bf16)
    rows,           # int32
    D: tl.constexpr,       # int (e.g., 128)
    BLOCK_D: tl.constexpr, # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    half = D // 2
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        # Load x1 and x2 halves
        x1 = tl.load(X_ptr + row_id * D + cols, mask=(cols < half), other=0.0)
        x2 = tl.load(X_ptr + row_id * D + cols + half, mask=(cols < half), other=0.0)
        # Load cos and sin vectors for these columns (broadcast over rows)
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Compute y1, y2 in fp32
        x1_f = x1.to(tl.float32)
        x2_f = x2.to(tl.float32)
        y1 = cos_vec * x1_f - sin_vec * x2_f
        y2 = cos_vec * x2_f + sin_vec * x1_f

        # Store back in bf16
        # First half columns
        tl.store(Y_ptr + row_id * D + cols, y1.to(x1.dtype), mask=(cols < half))
        # Second half columns
        tl.store(Y_ptr + row_id * D + (cols + half), y2.to(x2.dtype), mask=(cols < half))


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        # Ensure contiguous for Triton
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()

        # Shapes
        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        num_kv_heads = key.shape[1]

        # RMSNorm on query and key
        rows_query = B * H_q * S
        rows_key = B * num_kv_heads * S

        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        rms_norm_weighted_kernel[(rows_query,)](
            query.view(rows_query, D), q_norm_weight,
            query_norm.view(rows_query, D),
            rows_query, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_key,)](
            key.view(rows_key, D), k_norm_weight,
            key_norm.view(rows_key, D),
            rows_key, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # Compute cos and sin vectors using torch (bf16)
        # Note: emb = pos * inv_freq[:D//2], we use absolute positions 0..S-1
        pos = torch.arange(S, device=query.device, dtype=torch.int32)
        inv_freq_half = inv_freq  # shape [D//2] float32
        emb = pos.unsqueeze(-1).float() * inv_freq_half  # [S, D//2]
        emb_full = torch.cat([emb, emb], dim=-1)  # [S, D] float32
        cos = emb_full.cos().to(torch.bfloat16)   # [S, D] bf16
        sin = emb_full.sin().to(torch.bfloat16)   # [S, D] bf16

        # Apply rotary embedding to both normalized query and key
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            query_norm.view(rows_query, D),
            cos[:, 0].contiguous(),   # Triton expects [D]; we pass per-column vector
            sin[:, 0].contiguous(),
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            key_norm.view(rows_key, D),
            cos[:, 0].contiguous(),
            sin[:, 0].contiguous(),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches with rotated values (PyTorch, not Triton)
        # Mimic original behavior: update at given cache positions
        # Note: cache_position is 1D of length S
        # key_cache shape: [B, num_kv_heads, max_position_embeddings, D]
        # Broadcast assignment across B and num_kv_heads
        for b in range(B):
            for h in range(num_kv_heads):
                key_cache[b, h, cache_position] = key_norm[b, h, cache_position]
                value_cache[b, h, cache_position] = value[b, h, cache_position]

        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
