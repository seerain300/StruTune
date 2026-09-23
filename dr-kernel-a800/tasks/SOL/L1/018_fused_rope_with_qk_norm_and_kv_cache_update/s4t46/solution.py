import torch
import triton
import triton.language as tl


# Triton kernel: RMS normalization per row (length D).
# For each row, compute scale = 1/sqrt(mean(x^2) + eps), then y = x * scale.
@triton.jit
def rms_norm_rows_kernel(x_ptr, out_ptr, D: tl.constexpr, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = x * scale
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16))


# Triton kernel: copy/update cache at positions [cache_pos] for each (b, head, s).
# out_cache_ptr: destination cache (already allocated)
# src_ptr: source tensor (query_rotated or key_rotated for keys, value for values)
# B, num_heads, S are passed to derive the linear index: idx = b * (num_heads * D) + head * D
@triton.jit
def copy_update_cache_kernel(out_ptr, src_ptr, cache_pos, D: tl.constexpr, B, num_heads):
    pid_b = tl.program_id(0)
    pid_head = tl.program_id(1)
    s = tl.program_id(2)
    # compute source offset: (b * num_heads + head) * D + s * D
    # Note: src layout is [B, num_heads, S, D], so linearization: idx = b * (num_heads * D) + head * D + s * D
    src_offset = (pid_b * (num_heads * D) + pid_head * D + s * D)
    # destination offset: (b * num_heads + head) * D + cache_pos * D
    out_offset = (pid_b * (num_heads * D) + pid_head * D + cache_pos * D)
    x = tl.load(src_ptr + src_offset).to(tl.bfloat16)
    tl.store(out_ptr + out_offset, x)


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
        """
        Returns:
        - query_norm_rotated: query after RMS normalization and apply_rope
        - key_norm_rotated: key after RMS normalization and apply_rope
        - updated_key_cache: key_cache with rotated keys written at cache_position
        - updated_value_cache: value_cache with values written at cache_position
        """

        # 1) Triton RMS normalization for query
        B, num_q_heads, S, D = query.shape
        query_norm = torch.empty_like(query, dtype=torch.bfloat16, device=query.device)
        grid_q = (B * num_q_heads * S,)
        rms_norm_rows_kernel[grid_q](query, query_norm, D, rms_norm_eps)

        # 2) Triton RMS normalization for key (key shape: [B, num_kv_heads, S, D])
        Bk, num_kv_heads, Sk, _ = key.shape
        key_norm = torch.empty_like(key, dtype=torch.bfloat16, device=key.device)
        grid_k = (Bk * num_kv_heads * Sk,)
        rms_norm_rows_kernel[grid_k](key, key_norm, D, rms_norm_eps)

        # 3) Apply rotation (PyTorch, Triton lacks sin/cos): emb, cos, sin, rotate_half, and y = x*cos + rotate_half(x)*sin
        # Note: query_norm, key_norm are bfloat16
        # inv_freq: [half_head_dim] where half_head_dim = head_dim // 2 = 64
        half_dim = D // 2
        inv_freq = inv_freq.to(query_norm.dtype)

        position_ids_expanded = position_ids[:, :, None]  # [B, S, 1]
        emb = torch.cat([position_ids_expanded, position_ids_expanded], dim=-1).to(query_norm.dtype)  # [B, S, D]
        # Compute cos and sin on device using PyTorch
        cos = emb.cos()
        sin = emb.sin()

        # rotate_half: for x in [B, S, D], take x[..., :D/2], x[..., D/2:], return concat([-x2, x1])
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            return torch.cat([-x2, x1], dim=-1)

        query_rotated = query_norm * cos + rotate_half(query_norm) * sin
        key_rotated = key_norm * cos + rotate_half(key_norm) * sin

        # 4) Update caches using Triton kernel: write key_rotated to key_cache[:, :, cache_position]
        # key_cache shape: [B, num_kv_heads, max_len, D] = key_cache.shape
        # cache_position: [1, S], we interpret as a scalar per (b, head, s) at index cache_position[s]
        # We'll update for each (b, head, s)
        Bkv, num_kv_heads_v, max_len, D = key_cache.shape
        assert Bkv == B, "Batch size mismatch between query/key and key_cache"
        # Triton grid: (B, num_kv_heads, S)
        grid_kv = (B, num_kv_heads_v, Sk)
        # Note: cache_position is [1, S] int64 tensor; we take per-s lane value
        # We assume cache_len + s < max_len for given inputs (as in provided workloads)
        copy_update_cache_kernel[grid_kv](key_cache, key_rotated, cache_position[0], D, B, num_kv_heads_v)

        # Update value_cache: write value to value_cache[:, :, cache_position]
        # value shape: [B, num_kv_heads, S, D]
        Bv, num_kv_heads_v2, Sv, Dv = value.shape
        assert Bv == B and Sv == Sk and Dv == D, "Value shape mismatch"
        # Triton grid: (B, num_kv_heads, S)
        copy_update_cache_kernel[grid_kv](value_cache, value, cache_position[0], D, B, num_kv_heads_v)

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
