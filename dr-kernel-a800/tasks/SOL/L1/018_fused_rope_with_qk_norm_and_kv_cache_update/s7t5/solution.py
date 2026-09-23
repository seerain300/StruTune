import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr,         # *T input
    Y_ptr,         # *T output
    W_ptr,         # *T weight (length D), typically ones
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Accumulate sum of squares across D (fp32)
    sum_sq = 0.0
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        sum_sq += (x.to(tl.float32) * x.to(tl.float32))

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Apply weight and store (weight is 1D of length D; we assume generic but inputs use ones)
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        w = tl.load(W_ptr + i).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(Y_ptr + base + i * stride_d, y.to(x.dtype))


@triton.jit
def rotation_kernel(
    X_ptr,         # *T normalized input (query or key)
    Y_ptr,         # *T output rotated tensor
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    inv_freq_ptr,  # *fp32, length D//2
    use_cos,       # 1 -> query rotation (cos-based), 0 -> key rotation (sin-based)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    # Build angle: first half uses inv_freq, second half duplicates
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i] = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # Load x as vector across D
    idx = tl.arange(0, D)
    mask = idx < D
    x = tl.load(X_ptr + base + idx * stride_d, mask=mask, other=0.0)

    # Compute rotate_half(x) = [-x2, x1], where x1 = x[:D//2], x2 = x[D//2:]
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    if use_cos == 1:
        # query rotation: cos-based
        y = x * cos_vec - rot * sin_vec
    else:
        # key rotation: sin-based
        y = x * sin_vec - rot * cos_vec

    tl.store(Y_ptr + base + idx * stride_d, y.to(tl.float32), mask=mask)


@triton.jit
def scatter_update_cache_kernel(
    Y_ptr,         # *T tensor to write from (rotated keys or original values)
    Cache_ptr,     # *T cache tensor to update
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    cache_stride_b, cache_stride_h, cache_stride_seq, cache_stride_d,
    cache_position_ptr,  # *int32, length S
    D: tl.constexpr,
):
    # Each program handles one (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    for s in range(0, S):
        idx = tl.load(cache_position_ptr + s)  # int32
        base_in = pid_b * cache_stride_b + pid_h * cache_stride_h + s * cache_stride_seq
        base_out = pid_b * cache_stride_b + pid_h * cache_stride_h + idx * cache_stride_seq

        # Copy row: load from Y_ptr and store into Cache_ptr
        for d in range(0, D):
            val = tl.load(Y_ptr + base_in + d * cache_stride_d)
            tl.store(Cache_ptr + base_out + d * cache_stride_d, val.to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure device and dtype
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA."
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16, "Expected bf16 tensors."

        B, H_q, S, D = query.shape
        Bk, H_kv, Sk, Dk = key.shape
        assert Bk == B and Sk == S and Dk == D, "Key shape mismatch with query."

        # RMSNorm for query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Strides
        stride_b_q, stride_h_q, stride_s_q, stride_d_q = query.stride()
        stride_b_k, stride_h_k, stride_s_k, stride_d_k = key.stride()

        grid_q = (B, H_q, S)
        grid_k = (B, H_kv, S)

        rmsnorm_kernel[grid_q](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            stride_b_q, stride_h_q, stride_s_q, stride_d_q,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )
        rmsnorm_kernel[grid_k](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            stride_b_k, stride_h_k, stride_s_k, stride_d_k,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # Rotation: query uses cos, key uses sin
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        stride_b_qn, stride_h_qn, stride_s_qn, stride_d_qn = query_norm.stride()
        stride_b_kn, stride_h_kn, stride_s_kn, stride_d_kn = key_norm.stride()

        grid_qr = (B, H_q, S)
        grid_kr = (B, H_kv, S)

        inv_freq_fp32 = inv_freq.to(torch.float32)
        rotation_kernel[grid_qr](
            query_norm, query_rotated,
            B, H_q, S, D,
            stride_b_qn, stride_h_qn, stride_s_qn, stride_d_qn,
            inv_freq_fp32, 1,
            num_warps=4, num_stages=2
        )
        rotation_kernel[grid_kr](
            key_norm, key_rotated,
            B, H_kv, S, D,
            stride_b_kn, stride_h_kn, stride_s_kn, stride_d_kn,
            inv_freq_fp32, 0,
            num_warps=4, num_stages=2
        )

        # Update caches using Triton scatter
        key_cache_stride_b, key_cache_stride_h, key_cache_stride_seq, key_cache_stride_d = key_cache.stride()
        value_cache_stride_b, value_cache_stride_h, value_cache_stride_seq, value_cache_stride_d = value_cache.stride()

        cache_position_i32 = cache_position.to(torch.int32)

        # Scatter update for key cache
        grid_scatter_k = (B, H_kv)
        scatter_update_cache_kernel[grid_scatter_k](
            key_rotated,
            key_cache,
            B, H_kv, S,
            key_cache_stride_b, key_cache_stride_h, key_cache_stride_seq, key_cache_stride_d,
            cache_position_i32,
            D,
            num_warps=4, num_stages=2
        )

        # Scatter update for value cache (original code sets value_cache at the same positions)
        grid_scatter_v = (B, H_kv)
        scatter_update_cache_kernel[grid_scatter_v](
            value,
            value_cache,
            B, H_kv, S,
            value_cache_stride_b, value_cache_stride_h, value_cache_stride_seq, value_cache_stride_d,
            cache_position_i32,
            D,
            num_warps=4, num_stages=2
        )

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
