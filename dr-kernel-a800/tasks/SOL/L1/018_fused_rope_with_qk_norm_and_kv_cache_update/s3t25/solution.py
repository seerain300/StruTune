import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,            # *pointer* to input, shape [rows, D]
    W_ptr,            # *pointer* to weight vector, shape [D]
    Y_ptr,            # *pointer* to output, shape [rows, D]
    rows,             # int32
    D: tl.constexpr,             # int (e.g., 128)
    eps,                            # float32 scalar
    BLOCK_D: tl.constexpr,         # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    # Compute sum of squares over D
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
    X_ptr,            # *pointer* to input, shape [rows, D]
    Y_ptr,            # *pointer* to output, shape [rows, D]
    COS_ptr,          # *pointer* to cos vector, shape [D] float32
    SIN_ptr,          # *pointer* to sin vector, shape [D] float32
    rows,             # int32
    D: tl.constexpr,             # int (e.g., 128)
    BLOCK_D: tl.constexpr,       # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    # Process in tiles of BLOCK_D
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x1 = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x2 = tl.load(X_ptr + row_id * D + (cols + half), mask=mask, other=0.0)
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        # y1 = cos*x1 - sin*x2
        # y2 = cos*x2 + sin*x1
        y1 = cos_vec * x1.to(tl.float32) - sin_vec * x2.to(tl.float32)
        y2 = cos_vec * x2.to(tl.float32) + sin_vec * x1.to(tl.float32)
        y = tl.where(cols < half, y1, y2)  # [BLOCK_D]
        tl.store(Y_ptr + row_id * D + cols, y.to(x1.dtype), mask=mask)


@triton.jit
def _emb_cos_sin_kernel(
    POS_ptr,          # *pointer* to positions, shape [S] int32
    INV_ptr,          # *pointer* to inv_freq, shape [D//2] float32
    COS_ptr,          # *pointer* to output cos, shape [S, D//2] float32
    SIN_ptr,          # *pointer* to output sin, shape [S, D//2] float32
    S,                # int32
    D: tl.constexpr,             # int (e.g., 128)
):
    pos = tl.program_id(0)  # each program handles one position
    if pos >= S:
        return
    half = D // 2
    inv = tl.load(INV_ptr + tl.arange(0, half))  # [half]
    emb = tl.cast(pos, tl.float32) * inv         # [half]
    cos_val = tl.cos(emb)                        # [half]
    sin_val = tl.sin(emb)                        # [half]
    # store cos/sin to [pos, :]
    # COS_ptr is a flat buffer; SIN_ptr is a flat buffer.
    # We write [pos, 0..half-1]
    for i in range(0, half):
        tl.store(COS_ptr + pos * half + i, cos_val[i])
        tl.store(SIN_ptr + pos * half + i, sin_val[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()
        B, H_q, S, D = query.shape
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        H_k = key.shape[1]
        # Flatten rows for Triton
        rows_q = B * H_q * S
        rows_k = B * H_k * S

        # 1) RMSNorm on query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        rms_norm_weighted_kernel[(rows_q,)](
            query.view(rows_q, D),
            q_norm_weight,
            query_norm.view(rows_q, D),
            rows_q, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        rms_norm_weighted_kernel[(rows_k,)](
            key.view(rows_k, D),
            k_norm_weight,
            key_norm.view(rows_k, D),
            rows_k, D, float(rms_norm_eps), BLOCK_D=128, num_warps=4
        )

        # 2) Prepare cos/sin for apply_rope (float32)
        # position_ids shape [B, S], int64; we use S positions
        POS = position_ids[:, :].to(torch.int32).reshape(-1).contiguous()  # [S]
        S = POS.shape[0]
        half = D // 2
        inv_half = inv_freq[:half]  # [D//2] float32

        # Allocate cos/sin buffers [S, D//2] float32
        cos_half = torch.empty((S, half), dtype=torch.float32, device=query.device)
        sin_half = torch.empty((S, half), dtype=torch.float32, device=query.device)

        # Launch _emb_cos_sin_kernel: one program per position
        _emb_cos_sin_kernel[(S,)](
            POS, inv_half, cos_half, sin_half, S, D
        )

        # Expand to [S, D] for apply_rope: concat [cos_half, sin_half] along last dim
        cos = torch.empty((S, D), dtype=torch.float32, device=query.device)
        sin = torch.empty((S, D), dtype=torch.float32, device=query.device)
        cos[:, :half] = cos_half
        cos[:, half:] = sin_half
        sin[:, :half] = sin_half
        sin[:, half:] = cos_half

        # 3) Apply Rotary Embedding
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty((B, H_k, S, D), dtype=query_norm.dtype, device=query.device)

        apply_rope_kernel[(rows_q,)](
            query_norm.view(rows_q, D),
            query_rot.view(rows_q, D),
            cos[:, 0].contiguous(),   # cos is [S, D], we pass vector [D]
            sin[:, 0].contiguous(),   # sin is [S, D], we pass vector [D]
            rows_q, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_k,)](
            key_norm.view(rows_k, D),
            key_rot.view(rows_k, D),
            cos[:, 0].contiguous(),
            sin[:, 0].contiguous(),
            rows_k, D, BLOCK_D=128, num_warps=4
        )

        # 4) Update caches (PyTorch, non-compute)
        # Original code updates cache with rotated key. We return rotated tensors and update caches like original (though we don't use key_rot for output).
        key_cache[:, :, cache_position] = key_rot
        value_cache[:, :, cache_position] = value

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
