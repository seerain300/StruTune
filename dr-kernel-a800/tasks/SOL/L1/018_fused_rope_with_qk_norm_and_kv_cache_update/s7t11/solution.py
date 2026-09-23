import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Reduction across D
    sum_sq = 0.0
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        x_f = x.to(tl.float32)
        sum_sq += x_f * x_f

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + 1e-6)

    # Weight is 1D of length D; load scalar (weight is ones in provided get_inputs, but general code supports any weight)
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        w = tl.load(W_ptr + i).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(Y_ptr + base + i * stride_d, y.to(x.dtype))


@triton.jit
def rotation_kernel(
    X_ptr, Y_ptr, inv_freq_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_y_b, stride_y_h, stride_y_s, stride_y_d,
    use_cos: tl.constexpr,  # 1 -> query rotation (use cos/sin), 0 -> key rotation (use sin/cos)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    D_half = D // 2

    # Load the row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_ptr + base_x + i * stride_x_d)

    # Compute rotation scalars: for query a=cos(angle), b=sin(angle); for key a=sin(angle), b=cos(angle)
    if use_cos == 1:
        a = 0.0
        b = 0.0
    else:
        # keys: a = sin(angle), b = cos(angle)
        a = 0.0
        b = 0.0

    # angle[i] = s * inv_freq[i], i in [0, D_half)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle = pid_s * f
        a = angle  # sin
        b = angle  # cos
        # Note: For query, we need cos; for key, we need sin. The above initializes a/b. We will compute properly below.

    # Properly compute a and b depending on use_cos
    if use_cos == 1:
        # query rotation: a = cos(angle), b = sin(angle)
        for i in range(0, D_half):
            f = tl.load(inv_freq_ptr + i)
            angle = pid_s * f
            a = tl.cos(angle)
            b = tl.sin(angle)
            # Set the second half as duplicates for consistency, though we won't use it directly in rotation
            # We will use a and b when multiplying whole row, but Triton requires elementwise; so we apply per-element below.
            pass
        # Since we need per-element a/b for rotation, recompute directly per element in the next loop
        for i in range(0, D_half):
            f = tl.load(inv_freq_ptr + i)
            angle = pid_s * f
            a = tl.cos(angle)
            b = tl.sin(angle)
            # Rotate half
            x1 = x[:D_half]
            x2 = x[D_half:]
            rot = tl.cat([-x2, x1], axis=0)
            y = x * a - rot * b
            for k in range(0, D):
                tl.store(Y_ptr + base_y + k * stride_y_d, y[k].to(tl.float32))
    else:
        # key rotation: a = sin(angle), b = cos(angle)
        for i in range(0, D_half):
            f = tl.load(inv_freq_ptr + i)
            angle = pid_s * f
            a = tl.sin(angle)
            b = tl.cos(angle)
        # Per-element rotation
        for i in range(0, D_half):
            f = tl.load(inv_freq_ptr + i)
            angle = pid_s * f
            a = tl.sin(angle)
            b = tl.cos(angle)
            x1 = x[:D_half]
            x2 = x[D_half:]
            rot = tl.cat([-x2, x1], axis=0)
            y = x * a - rot * b
            for k in range(0, D):
                tl.store(Y_ptr + base_y + k * stride_y_d, y[k].to(tl.float32))


@triton.jit
def scatter_update_cache_kernel(
    SRC_ptr, DST_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    stride_src_b, stride_src_h, stride_src_s, stride_src_d,
    stride_dst_b, stride_dst_h, stride_dst_s, stride_dst_d,
    cache_pos_ptr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    idx = tl.load(cache_pos_ptr + pid_s)  # int32
    src_base = pid_b * stride_src_b + pid_h * stride_src_h + pid_s * stride_src_s
    dst_base = pid_b * stride_dst_b + pid_h * stride_dst_h + idx * stride_dst_s

    # Copy row SRC[b, h, s, :] to DST[b, h, idx, :]
    for i in range(0, D):
        val = tl.load(SRC_ptr + src_base + i * stride_src_d)
        tl.store(DST_ptr + dst_base + i * stride_dst_d, val.to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure tensors are on the same device
        device = query.device

        # 1) RMSNorm for query
        B, H_q, S, D = query.shape
        query_norm = torch.empty_like(query)
        stride_q_x = query.stride(0), query.stride(1), query.stride(2), query.stride(3)
        stride_q_y = query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3)
        grid_q = (B, H_q, S)
        rmsnorm_kernel[grid_q](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            stride_q_x[0], stride_q_x[1], stride_q_x[2], stride_q_x[3],
            num_warps=4,
        )

        # 2) RMSNorm for key
        B, H_kv, S, D = key.shape  # num_kv_heads=8
        key_norm = torch.empty_like(key)
        stride_k_x = key.stride(0), key.stride(1), key.stride(2), key.stride(3)
        stride_k_y = key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3)
        grid_k = (B, H_kv, S)
        rmsnorm_kernel[grid_k](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            stride_k_x[0], stride_k_x[1], stride_k_x[2], stride_k_x[3],
            num_warps=4,
        )

        # 3) Rotation using Triton kernels
        dtype = query.dtype  # bfloat16
        query_rot = torch.empty_like(query_norm)  # output rotated query
        key_rot = torch.empty_like(key_norm)      # output rotated key

        grid_qrot = (B, H_q, S)
        # For query, use_cos=1 (cos-based rotation)
        rotation_kernel[grid_qrot](
            query_norm, query_rot, inv_freq,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            1,
            num_warps=4,
        )

        grid_krot = (B, H_kv, S)
        # For key, use_cos=0 (sin-based rotation)
        rotation_kernel[grid_krot](
            key_norm, key_rot, inv_freq,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            0,
            num_warps=4,
        )

        # Cast back to original dtype
        query_rot = query_rot.to(dtype)
        key_rot = key_rot.to(dtype)

        # 4) Scatter update caches with rotated keys and original values
        L = key_cache.shape[2]
        grid_scatter = (B, H_kv, S)
        # Cast sources to bf16 for store (DST is bf16)
        scatter_update_cache_kernel[grid_scatter](
            key_rot.to(torch.bfloat16), key_cache,
            B, H_kv, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position.to(torch.int32),
            num_warps=4,
        )

        scatter_update_cache_kernel[grid_scatter](
            value.to(torch.bfloat16), value_cache,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position.to(torch.int32),
            num_warps=4,
        )

        # Return exactly what the original run() returns: query_rotated, key_rotated, key_cache, value_cache
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
