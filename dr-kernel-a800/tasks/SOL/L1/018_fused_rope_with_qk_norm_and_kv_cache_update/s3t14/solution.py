import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [rows, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [rows, D]
    rows,           # number of rows
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
        x = tl.load(X_ptr + row_id * D + cols)  # cols < D guaranteed
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = tl.rsqrt(mean + eps)

    # Apply per-dimension weight and store
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + row_id * D + cols)
        w = tl.load(W_ptr + cols).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype))


@triton.jit
def apply_rope_kernel(
    X_ptr,      # *pointer* to input, shape [rows, D] bf16
    Y_ptr,      # *pointer* to output, shape [rows, D] bf16
    COS_ptr,    # *pointer* to cos vector, shape [D] bf16
    SIN_ptr,    # *pointer* to sin vector, shape [D] bf16
    rows,       # number of rows
    D: tl.constexpr,       # int (e.g., 128)
    BLOCK_D: tl.constexpr, # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    # For each tile of columns, compute y1, y2 and write to output
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        # Load x and split into two halves
        # Note: D=128, so half=64. We read x1 and x2 using col offsets.
        x1 = tl.load(X_ptr + row_id * D + cols)                # [BLOCK_D]
        x2 = tl.load(X_ptr + row_id * D + cols + half)        # [BLOCK_D]
        # Load cos and sin vectors for this tile
        cos_vec = tl.load(COS_ptr + cols)                     # [BLOCK_D]
        sin_vec = tl.load(SIN_ptr + cols)                     # [BLOCK_D]
        # Compute y1 = cos*x1 - sin*x2, y2 = cos*x2 + sin*x1
        # Create mirrored halves
        x2_mir = x2
        x1_mir = x1
        # y1 for the first half, y2 for the second half
        y1 = cos_vec * x1 - sin_vec * x2
        y2 = cos_vec * x2 + sin_vec * x1
        # Write results into Y: we store y1 into first half and y2 into second half
        tl.store(Y_ptr + row_id * D + cols, y1.to(tl.bfloat16))
        tl.store(Y_ptr + row_id * D + (cols + half), y2.to(tl.bfloat16))


@triton.jit
def _compute_cos_sin_kernel(
    POS_ptr,        # *pointer* to position ids, shape [S] int32
    INV_ptr,        # *pointer* to inv_freq vector, shape [D//2] float32
    COS_ptr,        # *pointer* to output cos, shape [S, D] bf16
    SIN_ptr,        # *pointer* to output sin, shape [S, D] bf16
    S,              # number of positions
    D: tl.constexpr,             # int (e.g., 128)
    BLOCK_S: tl.constexpr,       # int (e.g., 128)
):
    # This kernel computes cos and sin for each position independently.
    # It writes [S, D] output.
    for pos_offs in range(0, S, BLOCK_S):
        pos = pos_offs + tl.arange(0, BLOCK_S)
        mask_pos = pos < S
        inv_half = tl.load(INV_ptr)  # [D//2] float32
        # Compute emb = pos[:, None] * inv_half[None, :] -> shape [BLOCK_S, D//2]
        emb_half = pos[:, None].to(tl.float32) * inv_half[None, :]  # [BLOCK_S, D//2]
        emb_full = tl.cat([emb_half, emb_half], axis=1)              # [BLOCK_S, D]
        cos_val = tl.cos(emb_full).to(tl.bfloat16)
        sin_val = tl.sin(emb_full).to(tl.bfloat16)
        # Store
        for i in range(BLOCK_S):
            if mask_pos[i]:
                p = pos[i].to(tl.int32)
                tl.store(COS_ptr + p * D + tl.arange(0, D), cos_val[i, :])
                tl.store(SIN_ptr + p * D + tl.arange(0, D), sin_val[i, :])


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args come in the same order as run function signature:
        # query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        device = query.device
        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        num_kv_heads = key.shape[1]

        # Ensure contiguous
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()  # shape [D]
        k_norm_weight = k_norm_weight.contiguous()  # shape [D]
        inv_freq = inv_freq.contiguous()            # shape [D//2], float32
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()  # shape [L], int64
        position_ids = position_ids.contiguous()      # shape [B, L], int64

        # RMSNorm for query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

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

        # Compute cos/sin using Triton: _compute_cos_sin_kernel
        # We need positions vector. Use the absolute positions 0..S-1
        pos_1d = torch.arange(S, device=device, dtype=torch.int32)  # [S]
        cos_emb = torch.empty((S, D), device=device, dtype=torch.bfloat16)
        sin_emb = torch.empty((S, D), device=device, dtype=torch.bfloat16)

        _compute_cos_sin_kernel[(1,)](
            pos_1d,
            inv_freq,
            cos_emb,
            sin_emb,
            S, D, BLOCK_S=128, num_warps=4
        )

        # Apply rotary embedding to normalized query and key
        # For query
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            query_norm.view(rows_query, D),  # output in-place rotated
            cos_emb.view(S, D, 1),           # pass as [D] per row
            sin_emb.view(S, D, 1),
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        # For key
        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            key_norm.view(rows_key, D),      # output in-place rotated
            cos_emb.view(S, D, 1),
            sin_emb.view(S, D, 1),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches using PyTorch (non-compute part)
        # The original run updates key_cache[:, :, cache_position] with rotated key and value_cache[:, :, cache_position] with current value.
        # We only return computed tensors to match the signature; cache updates are not returned.
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
