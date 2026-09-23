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
def _emb_cos_sin_kernel(
    POS_ptr,       # *pointer* to int32 positions, shape [S]
    INV_ptr,       # *pointer* to float32 inv_freq, shape [D//2]
    COS_ptr,       # *pointer* to bf16 cos, shape [S, D]
    SIN_ptr,       # *pointer* to bf16 sin, shape [S, D]
    S,             # int32
    D: tl.constexpr,           # int (e.g., 128)
):
    pos_id = tl.program_id(0)  # each program handles one position
    if pos_id >= S:
        return
    pos = tl.load(POS_ptr + pos_id).to(tl.float32)  # absolute position id, 0-based
    for j in range(0, D):
        inv = tl.load(INV_ptr + j // 2).to(tl.float32)  # inv_freq[j//2]
        emb = pos * inv
        c = tl.cos(emb).to(tl.bfloat16)
        s = tl.sin(emb).to(tl.bfloat16)
        tl.store(COS_ptr + pos_id * D + j, c)
        tl.store(SIN_ptr + pos_id * D + j, s)


@triton.jit
def apply_rope_kernel(
    X_ptr,       # *pointer* to input, contiguous, shape [rows, D]
    Y_ptr,       # *pointer* to output, contiguous, shape [rows, D]
    COS_ptr,     # *pointer* to cos vector, shape [D] bf16
    SIN_ptr,     # *pointer* to sin vector, shape [D] bf16
    rows,        # int32
    D: tl.constexpr,           # int (e.g., 128)
    BLOCK_D: tl.constexpr,     # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load x for first half (cols) and second half (cols + half)
        x1 = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X_ptr + row_id * D + (cols + half), mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin vectors for these columns
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        # Compute outputs for first half
        y1 = cos_vec * x1 - sin_vec * x2
        # Compute outputs for second half
        y2 = cos_vec * x2 + sin_vec * x1

        # Store y1 to first half positions and y2 to second half positions
        # Since offs is 0 (BLOCK_D=128, D=128), cols < D always holds; we write both halves.
        tl.store(Y_ptr + row_id * D + cols, y1.to(tl.bfloat16), mask=mask)
        tl.store(Y_ptr + row_id * D + (cols + half), y2.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,  # [B, S] int64
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,  # [S] int64 (ignored in Triton)
        q_norm_weight: torch.Tensor,   # [D] bfloat16
        k_norm_weight: torch.Tensor,   # [D] bfloat16
        inv_freq: torch.Tensor,        # [D//2] float32
        rms_norm_eps: float,
    ):
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        num_kv_heads = key.shape[1]
        assert key.shape[2] == S and value.shape[2] == S, "seq_len mismatch in key/value"

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

        # Compute cos and sin for apply_rope via Triton (bf16), using absolute positions
        # position_ids is [B, S], we will use the second dimension (S). Create POS as [S].
        # Absolute positions: 0, 1, ..., S-1
        pos = torch.arange(S, device=query.device, dtype=torch.int32)  # [S]

        S_eff = S
        inv_freq_half = inv_freq  # shape [D//2] float32

        # Allocate output buffers [S, D] bf16
        cos_out = torch.empty((S_eff, D), dtype=torch.bfloat16, device=query.device)
        sin_out = torch.empty((S_eff, D), dtype=torch.bfloat16, device=query.device)

        _emb_cos_sin_kernel[(S_eff,)](
            pos,
            inv_freq_half,
            cos_out,
            sin_out,
            S_eff, D,
            num_warps=4
        )

        # Apply rotary embedding to both normalized query and key
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            query_norm.view(rows_query, D),  # overwrite
            cos_out.view(D),                  # per-column cos
            sin_out.view(D),                  # per-column sin
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            key_norm.view(rows_key, D),      # overwrite
            cos_out.view(D),
            sin_out.view(D),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches (PyTorch assignment, not Triton compute)
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        key_cache[:, :, cache_position] = key_norm[:, :num_key_value_heads, cache_position]
        value_current = value[:, :num_key_value_heads, :, :]
        value_cache[:, :, cache_position] = value_current

        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
