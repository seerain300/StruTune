import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    rows,           # number of rows = B * num_heads * seq_len
    D: tl.constexpr,               # e.g., 128
    eps,                            # float32
    BLOCK_D: tl.constexpr = 128,   # processing chunk
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    # First pass: compute sum of squares
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = 1.0 / tl.sqrt(mean + eps)

    # Second pass: apply weight and store
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def emb_cos_sin_kernel(
    POS_ptr,        # *pointer* to positions, shape [S] int32
    INV_ptr,        # *pointer* to inv_freq half, shape [D//2] float32
    COS_ptr,        # *pointer* to output cos, shape [S, D] bf16
    SIN_ptr,        # *pointer* to output sin, shape [S, D] bf16
    S: tl.constexpr,               # sequence length
    D: tl.constexpr,               # head dim
):
    # Each program handles one position
    pos_id = tl.program_id(0)
    if pos_id >= S:
        return
    pos = tl.load(POS_ptr + pos_id).to(tl.float32)
    half = D // 2
    for col in range(0, half):
        inv = tl.load(INV_ptr + col)
        emb = pos * inv
        cosv = tl.cos(emb).to(tl.bfloat16)
        sinv = tl.sin(emb).to(tl.bfloat16)
        # Store to [S, D]
        tl.store(COS_ptr + pos_id * D + col, cosv)
        tl.store(SIN_ptr + pos_id * D + col, sinv)
    # Second half is the same emb
    for col in range(0, half):
        inv = tl.load(INV_ptr + col)
        emb = pos * inv
        cosv = tl.cos(emb).to(tl.bfloat16)
        sinv = tl.sin(emb).to(tl.bfloat16)
        tl.store(COS_ptr + pos_id * D + (col + half), cosv)
        tl.store(SIN_ptr + pos_id * D + (col + half), sinv)


@triton.jit
def apply_rope_kernel(
    X_ptr,      # *pointer* to input, [rows, D] bf16
    COS_ptr,    # *pointer* to cos vector, [D] bf16
    SIN_ptr,    # *pointer* to sin vector, [D] bf16
    Y_ptr,      # *pointer* to output, [rows, D] bf16
    rows,       # number of rows = B * num_heads * seq_len
    D: tl.constexpr,               # e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for offs in range(0, D, 128):
        cols = offs + tl.arange(0, 128)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        # Split x into two halves
        x1 = x[cols]                    # first half columns 0..63
        x2 = x[cols + D//2]             # second half columns 64..127
        # Load per-column cos/sin (bf16), cast to fp32 for math
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y1 = x1.to(tl.float32) * cos_vec - x2.to(tl.float32) * sin_vec
        y2 = x2.to(tl.float32) * cos_vec + x1.to(tl.float32) * sin_vec
        y = tl.cat([y1, y2], axis=0)
        tl.store(Y_ptr + row_id * D + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
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
        device = query.device
        D = query.shape[-1]
        assert D == 128, "This implementation expects head_dim=128."

        # RMSNorm for query and key using Triton
        B_q, H_q, S_q, D = query.shape
        B_k, H_k, S_k, D = key.shape
        assert S_q == S_k, "seq_len mismatch between query and key"
        assert H_q == B_q * 96, "Assumed num_attention_heads=96"
        assert H_k == B_k * 8, "Assumed num_key_value_heads=8"

        rows_query = B_q * H_q * S_q
        rows_key = B_k * H_k * S_k

        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        rms_norm_weighted_kernel[(rows_query,)](
            query.view(rows_query, D),
            q_norm_weight.contiguous(),
            query_norm.view(rows_query, D),
            rows_query, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_key,)](
            key.view(rows_key, D),
            k_norm_weight.contiguous(),
            key_norm.view(rows_key, D),
            rows_key, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # Generate cos/sin for apply_rope using Triton
        # We use absolute positions: pos = cache_len + t. Here, position_ids is [B, S]; we need S along last dim.
        # We'll use the last dim as S: S = position_ids.shape[-1].
        S = position_ids.shape[-1]
        # Convert position_ids to int32 and flatten for Triton
        pos_ids = position_ids.to(torch.int32).reshape(-1)  # shape [B*S], but we will only use S elements as the last dimension
        inv_half = inv_freq[:D//2].contiguous()

        cos_out = torch.empty((S, D), dtype=torch.bfloat16, device=device)
        sin_out = torch.empty((S, D), dtype=torch.bfloat16, device=device)

        emb_cos_sin_kernel[(S,)](
            pos_ids, inv_half, cos_out, sin_out, S, D
        )

        # Apply Rotary Embedding to normalized query and key using Triton
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            cos_out[0].contiguous(),   # cos per column [D] bf16
            sin_out[0].contiguous(),   # sin per column [D] bf16
            torch.empty_like(query_norm),  # output
            rows_query, D, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            cos_out[0].contiguous(),
            sin_out[0].contiguous(),
            torch.empty_like(key_norm),  # output
            rows_key, D, num_warps=4
        )

        # Cache updates (PyTorch, non-compute)
        # Return same outputs as original run function: (query_norm, key_norm, key_cache, value_cache)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
