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
    X_ptr,      # *pointer* to input, contiguous, shape [rows, D]
    Y_ptr,      # *pointer* to output, contiguous, shape [rows, D]
    COS_ptr,    # *pointer* to cos vector, shape [D], bf16
    SIN_ptr,    # *pointer* to sin vector, shape [D], bf16
    rows,       # int32
    D: tl.constexpr,        # int (e.g., 128)
    BLOCK_D: tl.constexpr,  # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Load cos and sin once per kernel (they are constant across rows in this setup).
    # cos/sin depend on absolute positions, not on tokens, so a single load is sufficient.
    cos_val = tl.load(COS_ptr + 0).to(tl.float32)
    sin_val = tl.load(SIN_ptr + 0).to(tl.float32)

    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        # Split into two halves
        half = D // 2
        x1 = x_fp32[:half]
        x2 = x_fp32[half:]
        # y1 = cos*x1 - sin*x2; y2 = cos*x2 + sin*x1
        y1 = cos_val * x1 - sin_val * x2
        y2 = cos_val * x2 + sin_val * x1
        y_fp32 = tl.concatenate([y1, y2], axis=0)  # stitch halves
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
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
        rms_norm_eps: float,
    ):
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()

        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."

        # Prepare RMSNorm outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # RMSNorm on query and key
        rows_query = B * H_q * S
        rows_key = B * key.shape[1] * S

        rms_norm_weighted_kernel[(rows_query,)](
            query.view(rows_query, D), q_norm_weight,
            query_norm.view(rows_query, D),
            rows_query, D, rms_norm_eps, BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_key,)](
            key.view(rows_key, D), k_norm_weight,
            key_norm.view(rows_key, D),
            rows_key, D, rms_norm_eps, BLOCK_D=128, num_warps=4
        )

        # Compute cos and sin for apply_rope (bf16), constant across rows in this setup
        # inv_freq is [D//2] float32; we need [D] embeddings by cat with itself.
        inv_freq_half = inv_freq  # shape [D//2]
        emb = torch.cat([inv_freq_half, inv_freq_half], dim=0).to(torch.float32)  # shape [D]
        pos = position_ids.view(B, S).to(torch.float32)  # [B, S]
        # emb: [D], pos: [B, S]; broadcast to [B, S, D]
        freqs = pos.unsqueeze(-1) * emb  # [B, S, D]
        cos = torch.cos(freqs).to(torch.bfloat16)  # [B, S, D]
        sin = torch.sin(freqs).to(torch.bfloat16)  # [B, S, D]

        # Apply rotary embedding to both query_norm and key_norm
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D), query_norm.view(rows_query, D),
            cos.view(1, D), sin.view(1, D),
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D), key_norm.view(rows_key, D),
            cos.view(1, D), sin.view(1, D),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        #


def run(*args):
    return ModelNew()(*args)
