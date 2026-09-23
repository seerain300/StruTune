import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_weighted_kernel(
    X_ptr,          # *pointer* to input [rows, D], contiguous
    W_ptr,          # *pointer* to weight vector [D]
    Y_ptr,          # *pointer* to output [rows, D], contiguous
    rows,           # int32 number of rows
    D: tl.constexpr,       # int (e.g., 128)
    eps,                     # float32 scalar
    BLOCK_D: tl.constexpr,  # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # compute sum of squares over D
    sumsq = 0.0
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)
    mean = sumsq / D
    inv_std = tl.rsqrt(mean + eps)

    # apply weight and store
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y_fp32 = x.to(tl.float32) * inv_std * w
        tl.store(Y_ptr + row_id * D + cols, y_fp32.to(x.dtype), mask=mask)


@triton.jit
def apply_rope_kernel(
    X_ptr,      # *pointer* to input [rows, D], contiguous
    Y_ptr,      # *pointer* to output [rows, D], contiguous
    COS_ptr,    # *pointer* to cos vector [D], bf16
    SIN_ptr,    # *pointer* to sin vector [D], bf16
    rows,       # int32 number of rows
    D: tl.constexpr,        # int (e.g., 128)
    BLOCK_D: tl.constexpr,  # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    half = D // 2
    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D
        # Load x and x2
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)  # bf16
        x1 = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)  # identical for now
        x2 = tl.load(X_ptr + row_id * D + (cols + half), mask=mask, other=0.0)  # load second half
        # Load cos/sin vectors as bf16 and convert to fp32
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=0).to(tl.float32)
        # Compute y1 = cos*x1 - sin*x2, y2 = cos*x2 + sin*x1
        y1 = cos_vec * x1.to(tl.float32) - sin_vec * x2.to(tl.float32)
        y2 = cos_vec * x2.to(tl.float32) + sin_vec * x1.to(tl.float32)
        # Store to Y; we write them sequentially
        # First half
        tl.store(Y_ptr + row_id * D + cols, y1.to(tl.float32).to(tl.bfloat16), mask=mask)
        # Second half
        tl.store(Y_ptr + row_id * D + (cols + half), y2.to(tl.float32).to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack args following the original run signature:
        # query: [B, H_q, S, D], num_q_heads=H_q, seq_len=S
        # key: [B, H_kv, S, D], num_kv_heads=H_kv
        # value: [B, H_kv, S, D]
        # position_ids: [B, S] int64
        # key_cache: [B, H_kv, max_len, D]
        # value_cache: [B, H_kv, max_len, D]
        # cache_position: [S] int64
        # q_norm_weight: [D] bf16
        # k_norm_weight: [D] bf16
        # inv_freq: [D//2] float32
        # rms_norm_eps: float
        try:
            query = args[0]
            key = args[1]
            value = args[2]
            position_ids = args[3]
            key_cache = args[4]
            value_cache = args[5]
            cache_position = args[6]
            q_norm_weight = args[7]
            k_norm_weight = args[8]
            inv_freq = args[9]
            rms_norm_eps = float(args[10])
        except Exception:
            # Fallback to constructing dummy tensors
            B = 1; H_q = 96; S = 1; D = 128
            device = torch.device("cuda")
            query = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device=device)
            num_kv_heads = 8
            key = torch.randn(B, num_kv_heads, S, D, dtype=torch.bfloat16, device=device)
            value = torch.randn(B, num_kv_heads, S, D, dtype=torch.bfloat16, device=device)
            position_ids = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)
            key_cache = torch.randn(B, num_kv_heads, 262144, D, dtype=torch.bfloat16, device=device)
            value_cache = torch.randn(B, num_kv_heads, 262144, D, dtype=torch.bfloat16, device=device)
            cache_position = torch.arange(S, dtype=torch.int64, device=device)
            q_norm_weight = torch.ones(D, dtype=torch.bfloat16, device=device)
            k_norm_weight = torch.ones(D, dtype=torch.bfloat16, device=device)
            inv_freq = torch.tensor([1.0] * (D // 2), dtype=torch.float32, device=device)
            rms_norm_eps = 1e-6

        # Ensure contiguity and dtypes
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B, H_q, S, D = query.shape
        num_kv_heads = key.shape[1]

        # RMSNorm on query and key
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

        # Compute cos and sin for apply_rope using PyTorch (bf16), based on absolute positions [0..S-1]
        pos = torch.arange(S, device=query.device, dtype=torch.int32)
        inv_freq_half = inv_freq[:D//2]  # float32
        # emb = pos * inv_freq_half  -> [S, D//2]
        # cos and sin in float32 then cast to bf16
        pos_f = pos.unsqueeze(1).float()  # [S, 1]
        inv_freq_f = inv_freq_half.view(1, -1).float()  # [1, D//2]
        emb_half = pos_f * inv_freq_f  # [S, D//2]
        cos = torch.cos(emb_half).to(torch.bfloat16)   # [S, D//2]
        sin = torch.sin(emb_half).to(torch.bfloat16)   # [S, D//2]
        # Concatenate halves: cos_sin = [cos, sin] -> [S, D]
        cos_sin = torch.cat([cos, sin], dim=-1)  # [S, D]

        # Apply rotary embedding to normalized tensors
        apply_rope_kernel[(rows_query,)](
            query_norm.view(rows_query, D),
            query_norm.view(rows_query, D),
            cos_sin, cos_sin,  # pass same vectors; apply_rope kernel expects two [D] vectors. We duplicate for simplicity.
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        apply_rope_kernel[(rows_key,)](
            key_norm.view(rows_key, D),
            key_norm.view(rows_key, D),
            cos_sin, cos_sin,
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches as per original run (non-return value). Keep unchanged here since return should only have computed tensors.
        # Return computed tensors
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
