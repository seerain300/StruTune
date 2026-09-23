import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm on a tensor X of shape [B, H, S, D]
# Y = (X * (1 / sqrt(mean(X^2) + eps))) * weight, where weight is 1D of length D
# We assume D is divisible by BLOCK_SIZE (here 128), and we handle general D by looping.
@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    W_ptr,  # weight, length D (dtype inferred from X)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    sum_sq = 0.0
    # reduction across D in chunks of BLOCK_SIZE
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + base + idx * stride_d, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_sq += tl.sum(x32 * x32, axis=0)

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # apply weight and store
    # Note: weight is 1D; we load a chunk at a time and apply elementwise
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + base + idx * stride_d, mask=mask, other=0.0)
        w = tl.load(W_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(Y_ptr + base + idx * stride_d, y.to(x.dtype), mask=mask)


# Triton kernel: apply rotation to normalized tensors using precomputed cos_all/sin_all
# For each (b, h, s):
# - query: y = x * cos - rotate_half(x) * sin
# - key:   y = x * sin - rotate_half(x) * cos
# cos_all/sin_all are tensors of shape [B, S, D] provided as pointers.
@triton.jit
def rotation_kernel(
    X_norm_ptr, Y_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    cos_all_ptr, sin_all_ptr,  # *fp32, shapes [B, S, D]
    use_cos: tl.constexpr,     # 1 for query rotation (use cos), 0 for key rotation (use sin)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Load x_norm row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_norm_ptr + base + i * stride_d).to(tl.float32)

    # Load cos/sin vectors for this (b, s)
    cos_vec = tl.zeros([D], dtype=tl.float32)
    sin_vec = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        cos_vec[i] = tl.load(cos_all_ptr + base + i * stride_d).to(tl.float32)
        sin_vec[i] = tl.load(sin_all_ptr + base + i * stride_d).to(tl.float32)

    # rotate_half(x) = [-x[D//2:], x[:D//2]]
    x1 = x[:D // 2]
    x2 = x[D // 2:]
    rot = tl.cat([-x2, x1], axis=0)

    if use_cos == 1:
        # query rotation: cos-based
        y = x * cos_vec - rot * sin_vec
    else:
        # key rotation: sin-based (original uses sin for keys)
        y = x * sin_vec - rot * cos_vec

    # store result
    for i in range(0, D):
        tl.store(Y_ptr + base + i * stride_d, y[i].to(tl.float32), mask=True)


# Triton kernel: scatter update caches: for each (b, h_kv, s), write rotated_key/value into cache at cache_position[s]
@triton.jit
def scatter_update_cache_kernel(
    src_ptr,            # pointer to rotated tensor or value tensor, shape [B, H, S, D]
    dest_ptr,           # pointer to key_cache or value_cache, shape [B, H, L, D]
    cache_pos_ptr,      # *int64, length S
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dest_stride_b, dest_stride_h, dest_stride_l, dest_stride_d,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    idx = tl.load(cache_pos_ptr + pid_s).to(tl.int32)

    src_base = pid_b * src_stride_b + pid_h * src_stride_h + pid_s * src_stride_s
    dest_base = pid_b * dest_stride_b + pid_h * dest_stride_h + idx * dest_stride_l

    # copy row of length D (store as fp32 to match cache dtype)
    for i in range(0, D):
        val = tl.load(src_ptr + src_base + i * src_stride_d).to(tl.float32)
        tl.store(dest_ptr + dest_base + i * dest_stride_d, val, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Inputs:
          query: [B, num_q_heads, S, D] (B, 96, S, 128), dtype bfloat16, device
          key:    [B, num_kv_heads, S, D] (B, 8, S, 128)
          value:  [B, num_kv_heads, S, D]
          position_ids: [B, S], int64
          key_cache: [B, num_kv_heads, L, D], dtype bfloat16
          value_cache: [B, num_kv_heads, L, D], dtype bfloat16
          cache_position: [S], int64
          q_norm_weight: [D], bfloat16 (ones)
          k_norm_weight: [D], bfloat16 (ones)
          inv_freq: [D//2], float32
          rms_norm_eps: float
        Returns:
          query_rotated: [B, num_q_heads, S, D]
          key_rotated:   [B, num_kv_heads, S, D]
          key_cache: updated [B, num_kv_heads, L, D]
          value_cache: updated [B, num_kv_heads, L, D]
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA for Triton."

        B_q, H_q, S, D = query.shape
        B_k, H_kv, S_k, D_k = key.shape
        assert B_q == B_k and S == S_k and D == D_k, "Input shapes must match expected dimensions."

        # Ensure contiguous layouts for predictable strides
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()

        # Output tensors for normalized inputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch RMSNorm kernels: one per (b, h, s)
        grid_norm = (B_q, H_q, S)
        rmsnorm_kernel[grid_norm](
            query, query_norm,
            B_q, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            q_norm_weight, rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4
        )

        grid_norm_k = (B_q, H_kv, S)
        rmsnorm_kernel[grid_norm_k](
            key, key_norm,
            B_q, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            k_norm_weight, rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4
        )

        # Precompute cos_all and sin_all on device using PyTorch to match original behavior.
        # inv_freq is length D//2; construct angles and then cos/sin.
        inv_half = D // 2
        pos = torch.arange(0, S, dtype=torch.float32, device=query.device).unsqueeze(1)  # [S, 1]
        angles = pos * inv_freq.view(1, inv_half)  # [S, D//2]
        cos_half = torch.cos(angles)  # [S, D//2]
        sin_half = torch.sin(angles)  # [S, D//2]
        # Duplicate across second half to form D-length vectors
        cos_all = torch.cat([cos_half, cos_half], dim=1).to(torch.float32)  # [S, D]
        sin_all = torch.cat([sin_half, sin_half], dim=1).to(torch.float32)  # [S, D]

        # Apply rotation via Triton kernels.
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        grid_rot = (B_q, H_q, S)
        rotation_kernel[grid_rot](
            query_norm, query_rotated,
            B_q, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            cos_all, sin_all,
            use_cos=1,  # query uses cos
            num_warps=4
        )

        grid_rot_k = (B_q, H_kv, S)
        rotation_kernel[grid_rot_k](
            key_norm, key_rotated,
            B_q, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            cos_all, sin_all,
            use_cos=0,  # key uses sin-based rotation
            num_warps=4
        )

        # Update caches via Triton scatter kernel
        B = B_q
        H = H_kv
        L = key_cache.shape[2]  # cache_len + seq_len, but we only write at cache_position indices
        grid_cache = (B, H, S)

        # key_cache and value_cache are [B, H, L, D] contiguous
        scatter_update_cache_kernel[grid_cache](
            key_rotated,  # rotated keys to write into key_cache
            key_cache,    # destination key_cache
            cache_position,  # positions
            B, H, S, D, L,
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            num_warps=1
        )

        # Update value_cache at the same positions with 'value' (no rotation, as in original)
        scatter_update_cache_kernel[grid_cache](
            value,        # source values
            value_cache,  # destination
            cache_position,
            B, H, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=1
        )

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
