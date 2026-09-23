import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row
# Each program handles one row (length D). Compute mean of squares, scale, and write normalized row.
@triton.jit
def rms_norm_rows_kernel(x_ptr, w_ptr, out_ptr, N_rows: tl.constexpr, D: tl.constexpr, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=offs < D, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = w * x * scale
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16), mask=offs < D)

# Triton kernel: rotate_half(x) -> out where x is [D], out is [D]
# For i in [0, D//2): out[2*i] = -x[2*i+1], out[2*i+1] = x[2*i]
# For i >= D//2: out[i] = 0
@triton.jit
def rotate_half_kernel(x_ptr, out_ptr, D: tl.constexpr):
    half = D // 2
    idx = tl.program_id(0)
    if idx < half:
        a = tl.load(x_ptr + 2 * idx)
        b = tl.load(x_ptr + 2 * idx + 1)
        tl.store(out_ptr + 2 * idx, -b)
        tl.store(out_ptr + 2 * idx + 1, a)
    else:
        tl.store(out_ptr + idx, 0.0)

# Triton kernel: apply_rope per row: y = x * cos + rotate_half(x) * sin
# x, cos, sin: [N_rows, D] flattened; each program handles one row.
@triton.jit
def apply_rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, N_rows: tl.constexpr, D: tl.constexpr, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)

    # Load as float32 for math
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    c = tl.load(cos_ptr + row_id * D + offs).to(tl.float32)
    s = tl.load(sin_ptr + row_id * D + offs).to(tl.float32)

    # Compute rotate_half(x) into tmp
    tmp = tl.zeros([D], dtype=tl.float32)
    half = D // 2
    for i in range(0, half):
        a = x[2 * i]
        b = x[2 * i + 1]
        tmp[2 * i] = -b
        tmp[2 * i + 1] = a
    # For i >= half, tmp[i] remains 0

    y = x * c + tmp * s
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16), mask=offs < D)

class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Inputs:
          - query: [B, 96, S, 128], bfloat16
          - key: [B, 8, S, 128], bfloat16
          - value: [B, 8, S, 128], bfloat16
          - position_ids: [B, S], int64
          - key_cache, value_cache: [B, 8, 262144, 128], bfloat16
          - cache_position: [S], int64
          - q_norm_weight, k_norm_weight: [128], bfloat16
          - inv_freq: [64], float32
          - rms_norm_eps: float
        Returns:
          - query_rotated: [B, 96, S, 128], bfloat16
          - key_rotated: [B, 8, S, 128], bfloat16
          - key_cache updated at positions cache_position
          - value_cache updated at positions cache_position
        """
        B, num_q_heads, S, head_dim = query.shape
        Bk, num_kv_heads, Sk, head_dim_k = key.shape
        assert head_dim == head_dim_k, "head_dim mismatch between query/key/value"
        D = head_dim
        assert D == 128, "This optimized kernel expects head_dim=128"
        D_half = D // 2
        assert len(inv_freq) == D_half, "inv_freq length must be head_dim//2"

        # Compute cos/sin for RoPE following original semantics:
        # emb = pos * inv_freq[:D_half], duplicated to full D.
        pos = position_ids[:, :, None].to(torch.float32)  # [B, S, 1]
        # Broadcast inv_freq to [B, S, D_half]
        inv_freq_broadcast = inv_freq[None, None, :].expand(1, S, D_half).to(torch.float32)
        emb = pos * inv_freq_broadcast  # [B, S, D_half]
        # Duplicate to full head_dim
        cos_emb = torch.cat([emb, emb], dim=-1)  # [B, S, D]
        sin_emb = torch.cat([emb, emb], dim=-1)  # [B, S, D]
        # Compute cos/sin in float32
        cos = torch.cos(cos_emb)  # [B, S, D], float32
        sin = torch.sin(sin_emb)  # [B, S, D], float32

        # RMS normalization using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch RMS normalization for query
        N_rows_q = B * num_q_heads * S
        rms_norm_rows_kernel[(N_rows_q,)](query, q_norm_weight, query_norm, N_rows_q, D, rms_norm_eps)

        # Launch RMS normalization for key
        N_rows_k = Bk * num_kv_heads * Sk  # Sk == S
        rms_norm_rows_kernel[(N_rows_k,)](key, k_norm_weight, key_norm, N_rows_k, D, rms_norm_eps)

        # Apply RoPE using Triton
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        # For query: each row is of length D, flattened over N_rows = B * num_q_heads * S
        N_rows_q = B * num_q_heads * S
        apply_rope_kernel[(N_rows_q,)](query_norm, cos, sin, query_rotated, N_rows_q, D, rms_norm_eps)

        # For key: N_rows = B * num_kv_heads * S
        N_rows_k = Bk * num_kv_heads * S
        apply_rope_kernel[(N_rows_k,)](key_norm, cos, sin, key_rotated, N_rows_k, D, rms_norm_eps)

        # Cache writes using PyTorch (data movement, not Triton)
        # Update key_cache at positions cache_position
        for b in range(B):
            key_cache[b, :, cache_position] = key_rotated[b]
        # Update value_cache: write current value at cache_position
        for b in range(B):
            value_cache[b, :, cache_position] = value[b]

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
