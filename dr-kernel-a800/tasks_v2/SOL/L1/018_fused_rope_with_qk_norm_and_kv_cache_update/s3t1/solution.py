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
    cos_val = tl.load(COS_ptr + 0).to(tl.float32)
    sin_val = tl.load(SIN_ptr + 0).to(tl.float32)

    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        half = D // 2
        x1 = x_fp32[:half]
        x2 = x_fp32[half:]
        y1 = cos_val * x1 - sin_val * x2
        y2 = cos_val * x2 + sin_val * x1
        y_fp32 = tl.concatenate([y1, y2], axis=0)
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def _emb_cos_sin_kernel(
    POS_ptr,      # *pointer* to int32 positions, shape [S]
    INV_ptr,      # *pointer* to float32 inv_freq, shape [D//2]
    COS_ptr,      # *pointer* to bf16 cos, shape [S, D]
    SIN_ptr,      # *pointer* to bf16 sin, shape [S, D]
    S,            # int32
    D: tl.constexpr,           # int (e.g., 128)
):
    pos_id = tl.program_id(0)  # each program handles one position
    if pos_id >= S:
        return
    pos = tl.load(POS_ptr + pos_id).to(tl.float32)  # absolute position id, 0-based
    for j in range(0, D):
        inv = tl.load(INV_ptr + j // 2).to(tl.float32)  # inv_freq[j//2]
        emb = pos * inv
        # Compute cos and sin in fp32, then cast to bf16
        c = tl.cos(emb).to(tl.bfloat16)
        s = tl.sin(emb).to(tl.bfloat16)
        tl.store(COS_ptr + pos_id * D + j, c)
        tl.store(SIN_ptr + pos_id * D + j, s)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,  # [B, S] int64
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,  # [S] int64 (not used in compute, but kept for signature)
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
        assert key.shape


def run(*args):
    return ModelNew()(*args)
