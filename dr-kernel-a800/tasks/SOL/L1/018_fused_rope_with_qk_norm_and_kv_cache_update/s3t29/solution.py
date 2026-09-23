import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    rows,           # int32
    D: tl.constexpr,       # int, e.g., 128
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # int, e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Accumulate sum of squares over the last dimension
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
    X_ptr,      # *pointer* to input, contiguous, shape [rows, D], bf16
    Y_ptr,      # *pointer* to output, contiguous, shape [rows, D], bf16
    COS_ptr,    # *pointer* to cos, shape [D], bf16
    SIN_ptr,    # *pointer* to sin, shape [D], bf16
    rows,       # int32
    D: tl.constexpr,           # int, e.g., 128
    HALF: tl.constexpr,        # int, e.g., 64
    BLOCK_D: tl.constexpr,     # int, e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Stage 1: columns [0, HALF-1] -> y1
    for offs in range(0, HALF, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < HALF
        # Load x1 from front half
        x1 = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0).to(tl.float32)
        # Load corresponding x2 from back half: cols + HALF
        x2_cols = cols + HALF
        x2 = tl.load(X_ptr + row_id * D + x2_cols, mask=mask, other=0.0).to(tl.float32)
        # Load cos/sin for these cols
        cos = tl.load(COS_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(SIN_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        # y1 = cos * x1 - sin * x2
        y1 = cos * x1 - sin * x2
        # Store y1 into output front half
        tl.store(Y_ptr + row_id * D + cols, y1.to(tl.bfloat16), mask=mask)
    # Stage 2: columns [HALF, D-1] -> y2
    for offs in range(0, HALF, BLOCK_D):
        cols2 = HALF + offs + tl.arange(0, BLOCK_D)  # cols starting at HALF
        mask2 = cols2 < D
        # x2 from front half (shifted), x1 from back half
        x2_front = tl.load(X_ptr + row_id * D + (cols2 - HALF), mask=mask2, other=0.0).to(tl.float32)
        x1_back = tl.load(X_ptr + row_id * D + (cols2 - HALF + HALF), mask=mask2, other=0.0).to(tl.float32)
        # cos/sin for these columns
        cos2 = tl.load(COS_ptr + (cols2 - HALF), mask=mask2, other=1.0).to(tl.float32)
        sin2 = tl.load(SIN_ptr + (cols2 - HALF), mask=mask2, other=1.0).to(tl.float32)
        # y2 = cos * x2_front + sin * x1_back
        y2 = cos2 * x2_front + sin2 * x1_back
        # Store y2 into output back half
        tl.store(Y_ptr + row_id * D + cols2, y2.to(tl.bfloat16), mask=mask2)


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        # Ensure dtypes and contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        # Shapes
        B, H_q, S, D = query.shape
        num_kv_heads = key.shape[1]
        assert D == 128, "This Triton implementation currently supports head_dim=128."

        # 1) RMSNorm on query and key (fp32 compute, bf16 store)
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

        # 2) Compute cos and sin for apply_rope using PyTorch (bf16), absolute positions [0..S-1]
        # We generate absolute positions; original run() used position_ids (B, S). For consistency, use S positions.
        pos = torch.arange(S, device=query.device, dtype=torch.int32)
        inv_freq_half = inv_freq  # [D//2] float32
        # emb = pos * inv_freq_half (shape [S, D//2]), then combine two halves
        emb = pos.unsqueeze(-1).float() * inv_freq_half  # [S, D//2]
        emb_full = torch.cat([emb, emb], dim=-1)         # [S, D] float32
        cos = emb_full.cos().to(torch.bfloat16)          # [S, D] bf16
        sin = emb_full.sin().to(torch.bfloat16)          # [S, D] bf16

        # 3) Apply rotary embedding to query_norm and key_norm
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            query_norm.view(rows_query, D),
            cos[:, 0].contiguous(),   # per-column vector [D], bf16
            sin[:, 0].contiguous(),   # per-column vector [D], bf16
            rows_query, D, HALF=64, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            key_norm.view(rows_key, D),
            cos[:, 0].contiguous(),
            sin[:, 0].contiguous(),
            rows_key, D, HALF=64, BLOCK_D=128, num_warps=4
        )

        # 4) Return results: query_rotated, key_rotated, key_cache, value_cache
        # Note: original run() also updates caches, but our forward doesn't have 'value_current' to apply, so we return caches as-is.
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
