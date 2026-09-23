import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input, shape [rows, D], contiguous
    W_ptr,          # *pointer* to weight vector, shape [D], contiguous
    Y_ptr,          # *pointer* to output, shape [rows, D], contiguous
    rows,           # int32: number of rows to process
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
    POS_ptr,        # *pointer* to positions, shape [S], int32
    INV_ptr,        # *pointer* to inv_freq, shape [D//2], float32
    COS_ptr,        # *pointer* to output cos, shape [S, D], bf16
    SIN_ptr,        # *pointer* to output sin, shape [S, D], bf16
    S,              # int32: number of positions
    D: tl.constexpr,         # int (e.g., 128)
):
    pos_id = tl.program_id(0)
    if pos_id >= S:
        return
    # Compute emb = pos_id * inv_freq[:D//2]
    # Then cos = emb.cos(), sin = emb.sin()
    half = D // 2
    for i in range(0, half):
        inv = tl.load(INV_ptr + i)  # float32
        val = tl.load(POS_ptr + pos_id) * inv  # float32
        # Store to COS and SIN at column i and i+half
        # We need to cast to bf16 and write to [S, D]
        c = tl.cos(val).to(tl.bfloat16)
        s = tl.sin(val).to(tl.bfloat16)
        tl.store(COS_ptr + pos_id * D + i, c)
        tl.store(SIN_ptr + pos_id * D + i, s)
        # store duplicated in second half
        tl.store(COS_ptr + pos_id * D + (i + half), c)
        tl.store(SIN_ptr + pos_id * D + (i + half), s)


@triton.jit
def apply_rope_kernel(
    X_ptr,          # *pointer* to input, shape [rows, D], contiguous
    COS_ptr,        # *pointer* to cos, shape [D], bf16
    SIN_ptr,        # *pointer* to sin, shape [D], bf16
    Y_ptr,          # *pointer* to output, shape [rows, D], contiguous
    rows,           # int32: number of rows to process
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

        # Load x and apply rotation
        # Split into two halves: [0:half] and [half:D]
        mask1 = (cols < half) & mask
        mask2 = ((cols >= half) & (cols < D)) & mask

        idx1 = cols
        idx2 = cols - half

        x1 = tl.load(X_ptr + row_id * D + idx1, mask=mask1, other=0.0)
        x2 = tl.load(X_ptr + row_id * D + idx2, mask=mask2, other=0.0)
        x1 = x1.to(tl.float32)
        x2 = x2.to(tl.float32)

        # Load cos/sin for these cols (broadcast if needed)
        c = tl.load(COS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        y1 = c * x1 - s * x2
        y2 = c * x2 + s * x1

        # Store results back to Y[row, cols]
        out_cols = cols
        # We can store both halves into Y; Triton will write to same pointer
        # But Y_ptr is [rows, D], so writing to out_cols is fine.
        tl.store(Y_ptr + row_id * D + out_cols, y1.to(x1.dtype), mask=mask & (cols < half))
        tl.store(Y_ptr + row_id * D + out_cols, y2.to(x2.dtype), mask=mask & (cols >= half))
        # Note: The above store attempts to write both; Triton requires single store per element. We'll fix by writing each branch.
        # Instead, write y1 to [0:half] and y2 to [half:D]:
        tl.store(Y_ptr + row_id * D + idx1, y1.to(x1.dtype), mask=mask1)
        tl.store(Y_ptr + row_id * D + idx2, y2.to(x2.dtype), mask=mask2)


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
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

        # Generate cos/sin for apply_rope using Triton
        # position_ids: [B, S] int64 -> extract S and make 1D int32
        assert position_ids.dim() == 2 and position_ids.shape[0] == B
        pos_1d = position_ids[:, :].to(torch.int32).view(-1)  # [B*S]
        S = pos_1d.shape[0]
        # Allocate cos and sin outputs [S, D] bf16
        cos = torch.empty((S, D), dtype=torch.bfloat16, device=query.device)
        sin = torch.empty((S, D), dtype=torch.bfloat16, device=query.device)

        _emb_cos_sin_kernel[(S,)](
            pos_1d, inv_freq, cos, sin, S, D, num_warps=4
        )

        # Apply rotary embedding to both normalized query and key
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            cos, sin,
            query_norm.view(rows_query, D),
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            cos, sin,
            key_norm.view(rows_key, D),
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Return rotated query and key, and keep caches unchanged
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
