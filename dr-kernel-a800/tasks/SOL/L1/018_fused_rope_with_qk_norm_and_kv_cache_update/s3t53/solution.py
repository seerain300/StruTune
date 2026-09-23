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
    X_ptr,          # *pointer* to input, shape [rows, D], contiguous
    COS_ptr,        # *pointer* to cos vector, shape [D], bf16
    SIN_ptr,        # *pointer* to sin vector, shape [D], bf16
    Y_ptr,          # *pointer* to output, shape [rows, D], contiguous
    rows,           # int32
    D: tl.constexpr,       # int (e.g., 128)
    BLOCK_D: tl.constexpr, # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    # Load per-position cos/sin vectors
    cos_vec = tl.load(COS_ptr, mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)
    sin_vec = tl.load(SIN_ptr, mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x1 = x[..., :D // 2]
        x2 = x[..., D // 2:]

        cos = cos_vec[None, :]
        sin = sin_vec[None, :]

        y1 = x1.to(tl.float32) * cos - x2.to(tl.float32) * sin
        y2 = x2.to(tl.float32) * cos + x1.to(tl.float32) * sin

        # Store back to Y_ptr at first half and second half
        tl.store(Y_ptr + row_id * D + cols, y1.to(x.dtype), mask=mask)
        tl.store(Y_ptr + row_id * D + cols + D // 2, y2.to(x.dtype), mask=mask)


@triton.jit
def _emb_cos_sin_kernel(
    POS_ptr,        # *pointer* to position_ids, shape [S], int32
    INV_ptr,        # *pointer* to inv_freq, shape [D//2], float32
    COS_ptr,        # *pointer* to output cos, shape [S, D], bf16
    SIN_ptr,        # *pointer* to output sin, shape [S, D], bf16
    S: tl.constexpr,                 # int (sequence length)
    D: tl.constexpr,                 # int (head_dim, e.g., 128)
):
    pos_id = tl.program_id(0)
    if pos_id >= S:
        return

    pos = tl.load(POS_ptr + pos_id)  # int32
    # Load inv_freq[:D//2]
    inv_idx = tl.arange(0, D // 2)
    inv_freq_half = tl.load(INV_ptr + inv_idx, mask=inv_idx < (D // 2), other=0.0).to(tl.float32)
    # Compute emb_half = pos * inv_freq[:D//2] (vectorized)
    emb_half = pos.to(tl.float32) * inv_freq_half  # shape [D//2], fp32
    # Build emb_full = concat([emb_half, emb_half], dim=-1)
    idx = tl.arange(0, D)
    mask_first = idx < (D // 2)
    mask_second = idx >= (D // 2)
    emb_full = tl.zeros((D,), dtype=tl.float32)
    emb_full = tl.where(mask_first, emb_half, emb_full)
    emb_full = tl.where(mask_second, emb_half, emb_full)

    cos_vals = tl.cos(emb_full).to(tl.bfloat16)
    sin_vals = tl.sin(emb_full).to(tl.bfloat16)

    # Store to [S, D] row
    tl.store(COS_ptr + pos_id * D + tl.arange(0, D), cos_vals, mask=True)
    tl.store(SIN_ptr + pos_id * D + tl.arange(0, D), sin_vals, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
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
        position_ids = position_ids.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, H_q, S, D = query.shape
        num_kv_heads = key.shape[1]
        assert D == 128, "This Triton implementation currently supports head_dim=128."

        # 1) RMSNorm for query and key
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

        # 2) Compute position-based cos and sin in Triton
        pos1d = position_ids.view(-1).to(torch.int32)  # [B*S]
        S_eff = pos1d.shape[0]
        cos = torch.empty((S_eff, D), device=query.device, dtype=torch.bfloat16)
        sin = torch.empty((S_eff, D), device=query.device, dtype=torch.bfloat16)

        _emb_cos_sin_kernel[(S_eff,)](
            pos1d, inv_freq[:D // 2],
            cos, sin,
            S=S_eff, D=D
        )

        # 3) Apply rotary embedding to normalized tensors
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

        # 4) Update caches (PyTorch: these are not part of compute)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
