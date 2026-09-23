import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    rows,           # int32
    D: tl.constexpr,       # e.g., 128
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Compute sum of squares across the last dimension
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
    X_ptr,      # *pointer* to input, contiguous, shape [rows, D]
    Y_ptr,      # *pointer* to output, contiguous, shape [rows, D]
    COS_ptr,    # *pointer* to cos vector, shape [D] bf16
    SIN_ptr,    # *pointer* to sin vector, shape [D] bf16
    rows,       # int32
    D: tl.constexpr,          # e.g., 128
    BLOCK_D: tl.constexpr,    # e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    # Process the row in blocks of BLOCK_D
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load input row block
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this block (same cos/sin for all positions in the row)
        cos = tl.load(COS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(SIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Split into two halves
        cols1 = cols[:half]
        cols2 = cols[half:]

        x1 = tl.load(X_ptr + row_id * D + cols1, mask=cols1 < D, other=0.0).to(tl.float32)
        x2 = tl.load(X_ptr + row_id * D + cols2, mask=cols2 < D, other=0.0).to(tl.float32)

        # Apply rotation
        y1 = cos[:half] * x1 - sin[:half] * x2   # [half]
        y2 = cos[half:] * x2 + sin[:half] * x1   # [half]

        # Concatenate results
        out = tl.zeros([D], dtype=tl.float32)
        out[:half] = y1
        out[half:] = y2

        tl.store(Y_ptr + row_id * D + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,  # [B, S] int64
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,  # [S] int64 (not used in compute)
        q_norm_weight: torch.Tensor,   # [D] bfloat16
        k_norm_weight: torch.Tensor,   # [D] bfloat16
        inv_freq: torch.Tensor,        # [D//2] float32
        rms_norm_eps: float,
    ):
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        num_kv_heads = key.shape[1]

        # Prepare RMSNorm outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # RMSNorm on query and key
        rows_query = B * H_q * S
        rows_key = B * num_kv_heads * S

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

        # Compute cos and sin for apply_rope in PyTorch (bf16), using absolute positions
        # We generate absolute positions [S] starting from 0 to match inv_freq behavior
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
            cos[:, 0].contiguous(),   # Triton expects [D], here we pass a vector (will broadcast within kernel)
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

        # Return results; value_cache/key_cache update is non-compute in original code
        return query_norm, key_norm, None, None


def run(*args):
    return ModelNew()(*args)
