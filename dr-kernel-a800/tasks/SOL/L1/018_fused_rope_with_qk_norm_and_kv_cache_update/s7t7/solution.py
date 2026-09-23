import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *T (bf16), input
    Y_ptr,        # *T (same as X dtype), output
    W_ptr,        # *T (same as X dtype), weight of length D
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,  # strides for X/Y
    eps,                                  # fp32 epsilon
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Accumulate sum of squares across D in fp32
    sum_sq = 0.0
    for i in range(0, D):
        xi = tl.load(X_ptr + base + i * stride_d)
        xi = xi.to(tl.float32)
        sum_sq += xi * xi

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Apply weight: w is 1D of length D
    for i in range(0, D):
        xi = tl.load(X_ptr + base + i * stride_d)
        wi = tl.load(W_ptr + i).to(tl.float32)
        yi = (xi.to(tl.float32) * inv_rms) * wi
        tl.store(Y_ptr + base + i * stride_d, yi.to(xi.dtype))


@triton.jit
def rotation_kernel(
    X_ptr,        # *T (normalized tensor), input
    Y_ptr,        # *T, output rotated tensor
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,  # strides for X/Y
    inv_freq_ptr,  # *fp32, length D//2
    use_cos,       # 1 -> query rotation (use cos), 0 -> key rotation (use sin)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    D_half = D // 2

    # Build angle vector across D: angle[i] = pid_s * inv_freq[i], duplicated across second half
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i] = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # Load x row (cast to fp32 for math)
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        xi = tl.load(X_ptr + base + i * stride_d)
        x[i] = xi.to(tl.float32)

    # Compute rotate_half(x): [-x2, x1], x1=x[:D//2], x2=x[D//2:]
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    if use_cos == 1:
        # query rotation: cos-based
        y = x * cos_vec - rot * sin_vec
    else:
        # key rotation: sin-based (original uses sin for keys), with cos duplicated across second half
        y = x * sin_vec - rot * cos_vec

    for i in range(0, D):
        tl.store(Y_ptr + base + i * stride_d, y[i].to(tl.float32))


@triton.jit
def scatter_update_cache_kernel(
    X_ptr,          # *T (rotated keys) or *T (values), shape [B, H_kv, S, D]
    KeyCache_ptr,   # *T, destination key_cache, shape [B, H_kv, L, D]
    ValueCache_ptr, # *T, destination value_cache, shape [B, H_kv, L, D]
    cache_pos_ptr,  # *int32, cache_position, shape [S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    stride_b_x, stride_h_x, stride_s_x, stride_d_x,  # strides for X (source)
    stride_b_k, stride_h_k, stride_l_k, stride_d_k,  # strides for KeyCache
    stride_b_v, stride_h_v, stride_l_v, stride_d_v,  # strides for ValueCache
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Loop over S and scatter to cache_position[s]
    for s in range(0, S):
        # Get cache index for this s
        idx = tl.load(cache_pos_ptr + s).to(tl.int32)

        # Source row base
        base_src = pid_b * stride_b_x + pid_h * stride_h_x + s * stride_s_x

        # Destination rows base
        base_k = pid_b * stride_b_k + pid_h * stride_h_k + idx * stride_l_k
        base_v = pid_b * stride_b_v + pid_h * stride_h_v + idx * stride_l_v

        # Copy across D
        for i in range(0, D):
            val = tl.load(X_ptr + base_src + i * stride_d_x)
            tl.store(KeyCache_ptr + base_k + i * stride_d_k, val)
            tl.store(ValueCache_ptr + base_v + i * stride_d_v, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure CUDA tensors
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda, "All tensors must be on CUDA."
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16, "Expected bf16 tensors for query/key/value."

        B, H_q, S, D = query.shape
        Bk, H_kv, Sk, Dk = key.shape
        assert Bk == B and Sk == S and Dk == D, "Key shape mismatch with query."

        # RMSNorm for query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        stride_b_q, stride_h_q, stride_s_q, stride_d_q = query.stride()
        stride_b_k, stride_h_k, stride_s_k, stride_d_k = key.stride()

        # Launch RMSNorm for query and key
        rmsnorm_kernel[(B, H_q, S)](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            stride_b_q, stride_h_q, stride_s_q, stride_d_q,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )
        rmsnorm_kernel[(B, H_kv, S)](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            stride_b_k, stride_h_k, stride_s_k, stride_d_k,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # Rotation: query uses cos, key uses sin (per original comment)
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        stride_b_qn, stride_h_qn, stride_s_qn, stride_d_qn = query_norm.stride()
        stride_b_kn, stride_h_kn, stride_s_kn, stride_d_kn = key_norm.stride()

        inv_freq_fp32 = inv_freq.to(torch.float32)
        rotation_kernel[(B, H_q, S)](
            query_norm, query_rotated,
            B, H_q, S, D,
            stride_b_qn, stride_h_qn, stride_s_qn, stride_d_qn,
            inv_freq_fp32, 1,  # use_cos=1 for query
            num_warps=4, num_stages=2
        )
        rotation_kernel[(B, H_kv, S)](
            key_norm, key_rotated,
            B, H_kv, S, D,
            stride_b_kn, stride_h_kn, stride_s_kn, stride_d_kn,
            inv_freq_fp32, 0,  # use_cos=0 for key (use sin)
            num_warps=4, num_stages=2
        )

        # Update caches: scatter rotated keys and values at cache_position indices
        cache_pos_i32 = cache_position.to(torch.int32)

        # Launch scatter kernel over (B, H_kv)
        scatter_update_cache_kernel[(B, H_kv)](
            key_rotated, key_cache, value_cache, cache_pos_i32,
            B, H_kv, S, D, key_cache.shape[2],
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=4, num_stages=2
        )

        # Return outputs consistent with original signature
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
