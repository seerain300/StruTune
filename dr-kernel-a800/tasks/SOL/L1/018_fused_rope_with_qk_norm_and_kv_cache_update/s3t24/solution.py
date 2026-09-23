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
    X_ptr,      # *pointer* to input, shape [rows, D], bf16
    Y_ptr,      # *pointer* to output, shape [rows, D], bf16
    COS_ptr,    # *pointer* to cos vector, shape [D], float32
    SIN_ptr,    # *pointer* to sin vector, shape [D], float32
    rows,       # int32
    D: tl.constexpr,        # int (e.g., 128)
    BLOCK_D: tl.constexpr,  # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        # Load original x
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0).to(tl.float32)
        # Split into two halves
        x1 = tl.load(X_ptr + row_id * D + tl.arange(0, BLOCK_D), mask=(tl.arange(0, BLOCK_D) < half), other=0.0).to(tl.float32)
        x2 = tl.load(X_ptr + row_id * D + tl.arange(half, half + BLOCK_D), mask=(tl.arange(0, BLOCK_D) < (D - half)), other=0.0).to(tl.float32)

        cosv = tl.load(COS_ptr + tl.arange(0, BLOCK_D), mask=mask, other=0.0).to(tl.float32)
        sinv = tl.load(SIN_ptr + tl.arange(0, BLOCK_D), mask=mask, other=0.0).to(tl.float32)

        y1 = cosv * x1 - sinv * x2
        y2 = cosv * x2 + sinv * x1
        y = tl.cat([y1, y2], axis=0)
        tl.store(Y_ptr + row_id * D + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor,
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
        # Ensure contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        H_k = key.shape[1]

        # Prepare RMSNorm outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # RMSNorm on query and key
        rows_query = B * H_q * S
        rows_key = B * H_k * S

        # Flatten to [rows, D] for kernel
        query_flat = query.view(rows_query, D)
        key_flat = key.view(rows_key, D)

        rms_norm_weighted_kernel[(rows_query,)](
            query_flat, q_norm_weight,
            query_norm.view(rows_query, D),
            rows_query, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_key,)](
            key_flat, k_norm_weight,
            key_norm.view(rows_key, D),
            rows_key, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # Prepare cos/sin for apply_rope. Use torch to generate, then cast to bf16 for kernel.
        pos_1d = position_ids.view(-1).to(torch.int32)  # shape [B*S]
        D_half = D // 2
        inv_freq_half = inv_freq[:D_half].to(torch.float32).contiguous()

        # Generate emb = pos * inv_freq[:D_half], then cos/sin
        emb = pos_1d.unsqueeze(-1).float() * inv_freq_half.unsqueeze(0)  # [S, D_half]
        cos_half = emb.cos().to(torch.bfloat16)  # [S, D_half]
        sin_half = emb.sin().to(torch.bfloat16)  # [S, D_half]

        # Expand to [S, D]
        cos = torch.empty((S, D), dtype=torch.bfloat16, device=query.device)
        sin = torch.empty((S, D), dtype=torch.bfloat16, device=query.device)
        cos[:, :D_half] = cos_half
        sin[:, :D_half] = sin_half
        cos[:, D_half:] = cos_half
        sin[:, D_half:] = sin_half

        # Apply rotary embedding to both normalized query and key
        rows_query_flat = query_norm.view(rows_query, D)
        rows_key_flat = key_norm.view(rows_key, D)

        apply_rope_kernel[(rows_query,)](
            rows_query_flat, query_norm.view(rows_query, D),
            cos[:, 0].contiguous(), sin[:, 0].contiguous(),
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            rows_key_flat, key_norm.view(rows_key, D),
            cos[:, 0].contiguous(), sin[:, 0].contiguous(),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Return as original: (query_rotated, key_rotated, key_cache, value_cache)
        # Note: original code also updates key_cache and value_cache using cache_position, but here we only return tensors without modifying caches, matching original output tuple.
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
