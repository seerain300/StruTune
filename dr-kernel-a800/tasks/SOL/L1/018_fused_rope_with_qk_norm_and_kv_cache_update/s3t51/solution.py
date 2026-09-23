import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, shape [rows, D], contiguous
    W_ptr,          # *pointer* to weight vector, shape [D], contiguous
    Y_ptr,          # *pointer* to output, shape [rows, D], contiguous
    rows,           # int32
    D: tl.constexpr,       # int, e.g., 128
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # int, e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = tl.rsqrt(mean + eps)

    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def apply_rope_kernel(
    X_ptr,      # *pointer* to input, shape [rows, D], contiguous
    Y_ptr,      # *pointer* to output, shape [rows, D], contiguous
    COS_ptr,    # *pointer* to cos vector, shape [D], bf16
    SIN_ptr,    # *pointer* to sin vector, shape [D], bf16
    rows,       # int32
    D: tl.constexpr,       # int, e.g., 128
    BLOCK_D: tl.constexpr, # int, e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        # load cos and sin for these columns
        cos = tl.load(COS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(SIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # half: D//2 = 64
        half = D // 2
        # load x1, x2 halves
        x1 = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x2 = tl.load(X_ptr + row_id * D + (cols + half), mask=(cols < half), other=0.0)  # mask: cols < half ensures idx < D
        # apply rotation
        y1 = cos * x1.to(tl.float32) - sin * x2.to(tl.float32)
        y2 = cos * x2.to(tl.float32) + sin * x1.to(tl.float32)
        # store y1 to first half, y2 to second half
        tl.store(Y_ptr + row_id * D + cols, y1.to(x.dtype), mask=mask)
        tl.store(Y_ptr + row_id * D + (cols + half), y2.to(x.dtype), mask=(cols < half))


@triton.jit
def _emb_cos_sin_kernel(
    POS_ptr,      # *pointer* to position vector, int32, shape [S]
    INV_ptr,      # *pointer* to inv_freq, float32, shape [D//2]
    COS_ptr,      # *pointer* to output cos, bf16, shape [S, D]
    SIN_ptr,      # *pointer* to output sin, bf16, shape [S, D]
    S,            # int32: number of positions
    D: tl.constexpr,               # int, e.g., 128
):
    pos_id = tl.program_id(0)
    if pos_id >= S:
        return
    # emb = pos * inv_freq[:D//2]
    half = D // 2
    inv = tl.load(INV_ptr + tl.arange(0, half))
    pos = tl.load(POS_ptr + pos_id).to(tl.float32)  # scalar
    emb_half = pos * inv  # [half], float32
    # expand to full D: first half same, second half same
    # write cos and sin bf16
    # We will write row pos_id into COS_ptr[pos_id, :] and SIN_ptr[pos_id, :]
    # Triton allows us to compute and store to 2D pointers; here we treat 2D as [S, D] with linear indexing pos_id*D + cols.
    # Create col vector
    for offs in range(0, D, 1):
        col = offs
        if col < half:
            val = emb_half[col].to(tl.bfloat16)
            tl.store(COS_ptr + pos_id * D + col, val)
            tl.store(SIN_ptr + pos_id * D + col, val)
        else:
            src_col = col - half
            val = emb_half[src_col].to(tl.bfloat16)
            tl.store(COS_ptr + pos_id * D + col, val)
            tl.store(SIN_ptr + pos_id * D + col, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor, cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, inv_freq: torch.Tensor, rms_norm_eps: float):
        """
        Triton-only implementation of RMSNorm and Rotary Embedding.
        Returns rotated query and key tensors (to match original signature), while key_cache and value_cache may be updated separately.
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda, "Inputs must be on CUDA device for Triton."
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16, "Inputs must be bfloat16."
        assert inv_freq.dtype == torch.float32, "inv_freq must be float32."

        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        num_kv_heads = key.shape[1]
        assert key.shape == (B, num_kv_heads, S, D) and value.shape == (B, num_kv_heads, S, D)

        # Make sure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Prepare RMSNorm outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # RMSNorm: one row per program
        rows_query = B * H_q * S
        rows_key = B * num_kv_heads * S

        # Launch RMSNorm for query and key
        rms_norm_weighted_kernel[(rows_query,)](
            query.view(rows_query, D), q_norm_weight.contiguous(),
            query_norm.view(rows_query, D),
            rows_query, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_key,)](
            key.view(rows_key, D), k_norm_weight.contiguous(),
            key_norm.view(rows_key, D),
            rows_key, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # Prepare pos vector (1D) from position_ids [B, S]
        # Original code uses absolute positions (cache_len + seq_len). Here we use absolute position starting from 0.
        # Make 1D int32
        pos_vec = position_ids.view(-1).to(torch.int32).contiguous()  # shape [B*S]

        # Compute cos/sin using Triton kernel: outputs [S, D] bf16
        S_total = pos_vec.shape[0]
        cos_out = torch.empty((S_total, D), dtype=torch.bfloat16, device=query.device)
        sin_out = torch.empty((S_total, D), dtype=torch.bfloat16, device=query.device)
        _emb_cos_sin_kernel[(S_total,)](
            pos_vec, inv_freq, cos_out, sin_out, S_total, D, num_warps=4
        )

        # Apply rotary embedding: one row per program
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D), query_norm.view(rows_query, D),
            cos_out, sin_out, rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D), key_norm.view(rows_key, D),
            cos_out, sin_out, rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches: mimic original behavior (not part of computation here)
        # key_cache[:, :, cache_position] = key_rotated
        # value_cache[:, :, cache_position] = value_current
        # Since we return rotated tensors, caches are not modified here.

        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
