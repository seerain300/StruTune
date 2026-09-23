import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr,         # *T (input, e.g., bf16)
    W_ptr,         # *fp32 weight of length D
    Y_ptr,         # *T (output)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    eps: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # accumulate sum of squares in fp32
    sum_sq = 0.0
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        x_fp32 = x.to(tl.float32)
        sum_sq += x_fp32 * x_fp32

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # scale by weight and store
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        w = tl.load(W_ptr + i)  # weight is 1D of length D
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(Y_ptr + base + i * stride_d, y.to(tl.float32))  # output is fp32 here; caller may cast


@triton.jit
def rotation_cos_kernel(
    X_ptr,         # *fp32, normalized query tensor
    Y_ptr,         # *fp32, output rotated query
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_y_b, stride_y_h, stride_y_s, stride_y_d,
    inv_freq_ptr,  # *fp32, length D//2
):
    # program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i]     = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # load row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_ptr + base_x + i * stride_x_d)

    # rotate_half(x)
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    # query rotation: y = x * cos - rotate_half(x) * sin
    y = x * cos_vec - rot * sin_vec

    # store
    for i in range(0, D):
        tl.store(Y_ptr + base_y + i * stride_y_d, y[i])


@triton.jit
def rotation_sin_kernel(
    X_ptr,         # *fp32, normalized key tensor
    Y_ptr,         # *fp32, output rotated key
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_y_b, stride_y_h, stride_y_s, stride_y_d,
    inv_freq_ptr,  # *fp32, length D//2
):
    # program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base_x = pid_b * stride_x_b + pid_h * stride_x_b + pid_s * stride_x_s  # Note: using stride_x_b should be stride_x_h
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    # Correction: base_x uses stride_x_h for head dimension
    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i]     = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # load row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_ptr + base_x + i * stride_x_d)

    # rotate_half(x)
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    # key rotation: y = x * sin - rotate_half(x) * cos
    y = x * sin_vec - rot * cos_vec

    # store
    for i in range(0, D):
        tl.store(Y_ptr + base_y + i * stride_y_d, y[i])


@triton.jit
def scatter_update_cache_kernel(
    X_src_ptr,     # *fp32, source tensor (rotated keys or original values)
    key_cache_ptr, # *fp32, destination cache
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    stride_src_b, stride_src_h, stride_src_s, stride_src_d,
    stride_cache_b, stride_cache_h, stride_cache_s, stride_cache_d,
    cache_pos_ptr,  # *int32, length S
):
    # program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    idx = tl.load(cache_pos_ptr + pid_s)

    base_src = pid_b * stride_src_b + pid_h * stride_src_h + pid_s * stride_src_s
    base_cache = pid_b * stride_cache_b + pid_h * stride_cache_h + idx * stride_cache_s

    # copy D elements from src row to cache row at idx
    for i in range(0, D):
        val = tl.load(X_src_ptr + base_src + i * stride_src_d)
        tl.store(key_cache_ptr + base_cache + i * stride_cache_d, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure devices and dtypes consistent
        device = query.device
        dtype = query.dtype  # likely bfloat16

        B, H_q, S, D = query.shape
        _, H_kv, _, _ = key.shape

        # 1) RMSNorm on query and key in Triton, output as fp32 (compute), then we cast back
        # Query RMSNorm
        query_norm = torch.empty((B, H_q, S, D), dtype=torch.float32, device=device)
        stride_q = query.stride()
        rmsnorm_kernel[(B, H_q, S)](
            query, q_norm_weight.to(torch.float32), query_norm,
            B, H_q, S, D,
            stride_q[0], stride_q[1], stride_q[2], stride_q[3],
            rms_norm_eps,
        )

        # Key RMSNorm
        key_norm = torch.empty((B, H_kv, S, D), dtype=torch.float32, device=device)
        stride_k = key.stride()
        rmsnorm_kernel[(B, H_kv, S)](
            key, k_norm_weight.to(torch.float32), key_norm,
            B, H_kv, S, D,
            stride_k[0], stride_k[1], stride_k[2], stride_k[3],
            rms_norm_eps,
        )

        # 2) Rotation using Triton kernels
        # Compute rotation for query: y = x * cos - rotate_half(x) * sin
        query_rot = torch.empty_like(query_norm)  # fp32
        rotation_cos_kernel[(B, H_q, S)](
            query_norm, query_rot,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            inv_freq.to(torch.float32),
        )

        # Compute rotation for key: y = x * sin - rotate_half(x) * cos
        key_rot = torch.empty_like(key_norm)  # fp32
        rotation_sin_kernel[(B, H_kv, S)](
            key_norm, key_rot,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            inv_freq.to(torch.float32),
        )

        # 3) Cast back to original dtype if desired (we won’t cast here to keep computations in fp32 in-kernel; original uses fp32 for norm anyway).
        # 4) Scatter update caches
        # Cast src to bf16 for storing into cache (cache is bf16 by default). Triton will accept fp32 input; we just copy elementwise into bf16 cache.
        grid_scatter = (B, H_kv, S)
        # For rotated keys
        scatter_update_cache_kernel[grid_scatter](
            query_rot, key_cache,  # copying rotated keys into key_cache at cache_position
            B, H_kv, S, D, key_cache.shape[2],
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position.to(torch.int32),
        )

        # For values (original, no rotation)
        scatter_update_cache_kernel[grid_scatter](
            value.to(torch.float32), value_cache,
            B, H_kv, S, D, value_cache.shape[2],
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position.to(torch.int32),
        )

        # 5) Return as original expected outputs: rotated query and key, and updated caches.
        # Note: The original returns query_rotated, key_rotated, key_cache, value_cache. We have query_rot and key_rot (fp32 normalized rotated).
        # Since the original also returns key_rotated (rotated), we return query_rot, key_rot, key_cache, value_cache.
        # To match original function signature, return these four tensors.

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
