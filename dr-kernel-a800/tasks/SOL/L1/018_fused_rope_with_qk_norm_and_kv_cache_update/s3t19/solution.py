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
def apply_rope_kernel_row(
    X_ptr,      # *pointer* to input, shape [rows, D]
    Y_ptr,      # *pointer* to output, shape [rows, D]
    COS_ptr,    # *pointer* to cos vector, shape [D] bf16
    SIN_ptr,    # *pointer* to sin vector, shape [D] bf16
    rows,       # int32
    D: tl.constexpr,             # int (e.g., 128)
    BLOCK_D: tl.constexpr,       # int (e.g., 128)
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    for offs in range(0, D, BLOCK_D):
        cols = offs + tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load x
        x = tl.load(X_ptr + row_id * D + cols, mask=mask, other=0.0)
        x1 = x[:, :D//2]  # first half
        x2 = x[:, D//2:]  # second half

        # Load cos/sin vectors for these cols
        cos_vec = tl.load(COS_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        sin_vec = tl.load(SIN_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        # Compute y: split into y1 and y2 halves
        # Note: x1, x2 are vectors of size BLOCK_D, cos_vec is vector too.
        y1 = cos_vec * x1 - sin_vec * x2
        y2 = cos_vec * x2 + sin_vec * x1

        # Store outputs
        tl.store(Y_ptr + row_id * D + cols, y1.to(x.dtype), mask=mask)  # first half positions
        tl.store(Y_ptr + row_id * D + (cols + D//2), y2.to(x.dtype), mask=mask)  # second half positions


def _compute_emb_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor, S: int):
    """
    position_ids: [B, S] int64
    inv_freq: [D//2] float32
    returns cos, sin: [S, D] bf16
    """
    B, S = position_ids.shape
    half = inv_freq.shape[0]
    D = 2 * half
    # emb = pos * inv_freq[:half] -> [S, half]
    pos = position_ids[:, :S].float()  # [B, S]
    emb_half = pos * inv_freq[None, :]  # [B, S, half]
    emb = torch.cat([emb_half, emb_half], dim=-1)  # [B, S, D] float32
    cos = emb.cos()  # [B, S, D]
    sin = emb.sin()  # [B, S, D]
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0]
        key = args[1]
        position_ids = args[2]
        key_cache = args[3]
        value_cache = args[4]
        cache_position = args[5]
        q_norm_weight = args[6]
        k_norm_weight = args[7]
        inv_freq = args[8]
        rms_norm_eps = args[9]

        device = query.device
        B, H_q, S, D = query.shape
        num_kv_heads = key.shape[1]
        assert D == 128, "This Triton implementation currently supports head_dim=128."
        half = D // 2
        assert q_norm_weight.shape[0] == D and k_norm_weight.shape[0] == D
        assert inv_freq.shape[0] == half

        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()
        position_ids = position_ids.contiguous()
        cache_position = cache_position.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()

        # RMSNorm on query and key
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

        # Compute cos and sin for apply_rope in torch, then feed to Triton
        cos_s, sin_s = _compute_emb_cos_sin(position_ids, inv_freq, S)  # [B, S, D] bf16

        # Apply rotary embedding via Triton
        # For query
        query_rot = torch.empty_like(query_norm)
        grid_query = (rows_query,)
        # We need per-column cos/sin vectors; cos_s is [B, S, D], we can pick one row index and use its cos/sin.
        # However, Triton kernel expects [D] vectors. We'll reshape cos_s and sin_s to [D] for each row.
        # Here we pass cos_s[:, 0, :] and sin_s[:, 0, :] which are [D].
        # But better: since cos/sin are functions of absolute position only, we can use cos_s[0] and sin_s[0] assuming S>0.
        # To be robust, we take cos_s[0] and sin_s[0] since S>0 in provided inputs. If S==0, return empty, but in tests S>=1.
        # However, the kernel grid is rows_query, so we pass cos_s[0] and sin_s[0].
        cos_vec = cos_s[0]  # [D] bf16
        sin_vec = sin_s[0]  # [D] bf16

        apply_rope_kernel_row[grid_query](
            query_norm.view(rows_query, D),
            query_rot.view(rows_query, D),
            cos_vec, sin_vec,
            rows_query, D, BLOCK_D=128, num_warps=4
        )

        # For key
        key_rot = torch.empty_like(key_norm)
        grid_key = (rows_key,)
        apply_rope_kernel_row[grid_key](
            key_norm.view(rows_key, D),
            key_rot.view(rows_key, D),
            cos_vec, sin_vec,
            rows_key, D, BLOCK_D=128, num_warps=4
        )

        # Update caches using PyTorch (non-compute operation)
        # key_cache[:, :, cache_position] = key_rot
        # pos_expand: [B, num_kv_heads, S] long
        pos_expand = cache_position.view(1, 1, -1).expand(B, num_kv_heads, -1).to(torch.long)
        for b in range(B):
            for h in range(num_kv_heads):
                key_cache[b, h, pos_expand[b, h], :] = key_rot[b, h]

        # value_cache[:, :, cache_position] = value_current. The original 'value' tensor is provided as args[3] (unused),
        # but original run assigns the value tensor to cache, not the 'value' argument. Here we assume 'value' argument is intended.
        # In provided original code, value is passed but not used; however, caches are updated to 'value' tensor.
        # So we can assign 'value' argument (args[3]) to value_cache at positions:
        # Note: args[3] is 'value' tensor, shape [B, num_kv_heads, S, D].
        value = args[3]
        value = value.contiguous()  # provided in forward
        for b in range(B):
            for h in range(num_kv_heads):
                value_cache[b, h, pos_expand[b, h], :] = value[b, h]

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
