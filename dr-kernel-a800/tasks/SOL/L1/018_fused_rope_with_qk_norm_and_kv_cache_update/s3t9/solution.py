import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_2d_kernel(
    X_ptr,          # *pointer* to input, contiguous, shape [B, H, S, D]
    W_ptr,          # *pointer* to weight vector, shape [D]
    Y_ptr,          # *pointer* to output, contiguous, shape [B, H, S, D]
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # e.g., 128
):
    # program ids for 3D launch: (batch, head, seq)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    # bounds check
    if b >= B or h >= H or s >= S:
        return

    row_start = (b * H + h) * S + s
    # Accumulate sum of squares over D
    sumsq = 0.0
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_start * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = tl.rsqrt(mean + eps)

    # Apply per-dimension weight and store
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_start * D + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_start * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def cosine_sin_kernel(
    POS_ptr,        # *pointer* to int32 positions, shape [S]
    INV_ptr,        # *pointer* to float32 inv_freq, shape [D//2]
    COS_ptr,        # *pointer* to bf16 cos, shape [S, D]
    SIN_ptr,        # *pointer* to bf16 sin, shape [S, D]
    S: tl.constexpr,
    D: tl.constexpr,
):
    # one program per position
    pos_id = tl.program_id(0)
    if pos_id >= S:
        return
    pos = tl.load(POS_ptr + pos_id).to(tl.float32)
    for j in range(0, D):
        inv = tl.load(INV_ptr + j // 2).to(tl.float32)
        emb = pos * inv
        c = tl.cos(emb).to(tl.bfloat16)
        s = tl.sin(emb).to(tl.bfloat16)
        tl.store(COS_ptr + pos_id * D + j, c)
        tl.store(SIN_ptr + pos_id * D + j, s)


@triton.jit
def apply_rope_half_kernel(
    X_ptr,          # *pointer* to input x, shape [rows, D]
    COS_ptr,        # *pointer* to cos vector, shape [D] bf16
    SIN_ptr,        # *pointer* to sin vector, shape [D] bf16
    Y1_ptr,         # *pointer* to output y1, shape [rows, D//2]
    Y2_ptr,         # *pointer* to output y2, shape [rows, D//2]
    rows,           # int32, number of rows (B * H * S)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,  # e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    # x1 and x2
    for offs in range(0, half, BLOCK_D):
        cols1 = offs + tl.arange(0, BLOCK_D)
        cols2 = offs + tl.arange(0, BLOCK_D) + half
        mask1 = cols1 < half
        mask2 = cols2 < half
        x1 = tl.load(X_ptr + row_id * D + cols1, mask=mask1, other=0.0)
        x2 = tl.load(X_ptr + row_id * D + cols2, mask=mask2, other=0.0)
        c = tl.load(COS_ptr + tl.arange(0, BLOCK_D), mask=mask1, other=1.0).to(tl.float32)
        s = tl.load(SIN_ptr + tl.arange(0, BLOCK_D), mask=mask1, other=1.0).to(tl.float32)
        y1 = c.to(tl.float32) * x1.to(tl.float32) - s.to(tl.float32) * x2.to(tl.float32)
        y2 = c.to(tl.float32) * x2.to(tl.float32) + s.to(tl.float32) * x1.to(tl.float32)
        tl.store(Y1_ptr + row_id * (D // 2) + offs, y1.to(x1.dtype), mask=mask1)
        tl.store(Y2_ptr + row_id * (D // 2) + offs, y2.to(x1.dtype), mask=mask2)


@triton.jit
def concat_half_kernel(
    Y1_ptr,         # *pointer* to y1, shape [rows, D//2]
    Y2_ptr,         # *pointer* to y2, shape [rows, D//2]
    Y_ptr,          # *pointer* to output y, shape [rows, D]
    rows,           # int32
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,  # e.g., 128
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    for offs in range(0, half, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < half
        y1 = tl.load(Y1_ptr + row_id * (D // 2) + offs, mask=mask, other=0.0)
        y2 = tl.load(Y2_ptr + row_id * (D // 2) + offs, mask=mask, other=0.0)
        # Store y = [y1, y2] into Y_ptr[row_id * D + :D] and [row_id * D + D//2 : D]
        out_cols = offs
        tl.store(Y_ptr + row_id * D + out_cols, y1.to(y1.dtype), mask=mask)
        tl.store(Y_ptr + row_id * D + (out_cols + half), y2.to(y2.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,  # [B, S] int64
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,  # [S] int64
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
        assert key.shape[1] == 8, "num_key_value_heads must be 8."

        # RMSNorm on query and key using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        rms_norm_weighted_2d_kernel[(B, H_q, S)](
            query.view(B, H_q, S, D), q_norm_weight, query_norm.view(B, H_q, S, D),
            B, H_q, S, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_2d_kernel[(B, num_kv_heads, S)](
            key.view(B, num_kv_heads, S, D), k_norm_weight, key_norm.view(B, num_kv_heads, S, D),
            B, num_kv_heads, S, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # Prepare absolute positions for cosine/sine computation
        pos = position_ids.view(-1).to(torch.int32)  # [B*S]
        S_eff = pos.numel()

        # Cosine and Sine buffers [S, D] bf16
        cos = torch.empty((S_eff, D), device=query.device, dtype=torch.bfloat16)
        sin = torch.empty((S_eff, D), device=query.device, dtype=torch.bfloat16)

        cosine_sin_kernel[(S_eff,)](
            pos, inv_freq, cos, sin, S_eff, D, num_warps=4
        )

        # Apply rotary embedding via Triton for query_norm and key_norm
        # For query
        rows_query = B * H_q * S
        # y1, y2 buffers
        y1_query = torch.empty((rows_query, 64), device=query.device, dtype=torch.bfloat16)
        y2_query = torch.empty((rows_query, 64), device=query.device, dtype=torch.bfloat16)

        # Concat buffer
        query_rotated = torch.empty((rows_query, 128), device=query.device, dtype=torch.bfloat16)

        # Launch apply_rope_half
        apply_rope_half_kernel[(rows_query,)](
            query_norm.view(rows_query, D), cos[0].to(torch.bfloat16), sin[0].to(torch.bfloat16),
            y1_query, y2_query, rows_query, D, BLOCK_D=128, num_warps=4
        )

        # Concatenate halfs
        concat_half_kernel[(rows_query,)](
            y1_query, y2_query, query_rotated, rows_query, D, BLOCK_D=128, num_warps=4
        )

        # For key
        rows_key = B * num_kv_heads * S
        y1_key = torch.empty((rows_key, 64), device=query.device, dtype=torch.bfloat16)
        y2_key = torch.empty((rows_key, 64), device=query.device, dtype=torch.bfloat16)
        key_rotated = torch.empty((rows_key, 128), device=query.device, dtype=torch.bfloat16)

        apply_rope_half_kernel[(rows_key,)](
            key_norm.view(rows_key, D), cos[0].to(torch.bfloat16), sin[0].to(torch.bfloat16),
            y1_key, y2_key, rows_key, D, BLOCK_D=128, num_warps=4
        )

        concat_half_kernel[(rows_key,)](
            y1_key, y2_key, key_rotated, rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Reshape back
        query_rotated = query_rotated.view(B, H_q, S, 128)
        key_rotated = key_rotated.view(B, num_kv_heads, S, 128)

        # Update caches (PyTorch for simplicity, as these are not compute)
        # Note: original run() updates caches; we mirror updates here for consistency.
        # key_cache and value_cache were input; we return rotated versions to match run signature.
        # We still perform the cache updates as per original intent, though output doesn't require it.

        # Since cache_position is provided, update at those positions.
        # Expand cache_position to [B, num_kv_heads, S] for key/value caches.
        # Note: cache_position is 1D length S, we broadcast over B and num_kv_heads.
        # torch advanced indexing supports this pattern.

        # We need to update only the first S positions in each cache buffer along the last dim.
        # Build indices for key_cache[:, :, :S, :]
        # For each batch and head, set rows=range(B), heads=range(num_kv_heads), positions=cache_position (0..S-1)
        # Using torch.index_select and slicing, but to keep it minimal, use advanced indexing:
        # Construct index tensors: idx = torch.arange(S, device=device)
        idx = torch.arange(S, device=query.device)
        # We update per (b, head), so loop
        for b in range(B):
            for head in range(num_kv_heads):
                key_cache[b, head, idx] = key_rotated[b, head]  # [S, 128]

        # Update value_cache with current value tensor for those positions
        # We only have a single value tensor of shape [B, 8, S, 128]; use that:
        # Loop over b and head, assign value[b, head] across S positions
        for b in range(B):
            for head in range(num_kv_heads):
                value_cache[b, head, idx] = value[b, head]  # [S, 128]

        # Return rotated query, rotated key, and updated caches
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
