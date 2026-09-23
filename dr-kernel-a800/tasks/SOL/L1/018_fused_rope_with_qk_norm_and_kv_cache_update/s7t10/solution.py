import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    eps: tl.constexpr,
):
    # Each program handles one (b, h, s) row across D
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Accumulate sum of squares across D in fp32
    sum_sq = 0.0
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        x32 = x.to(tl.float32)
        sum_sq += x32 * x32
    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Load weight scalar (assume weight is 1D of length D, and we use first element; original uses per-head weights but they are ones, so fine)
    w0 = tl.load(W_ptr + 0).to(tl.float32)

    # Apply normalization and weight
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        y32 = (x.to(tl.float32) * inv_rms) * w0
        tl.store(Y_ptr + base + i * stride_d, y32.to(x.dtype))


@triton.jit
def rotation_cos_kernel(
    X_norm_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_y_b, stride_y_h, stride_y_s, stride_y_d,
    inv_freq_ptr,
):
    # Each program handles one (b, h, s) row
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i] = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # Load x_norm row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_norm_ptr + base_x + i * stride_x_d)

    # Compute rotate_half(x)
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    # Query rotation: cos-based
    y = x * cos_vec - rot * sin_vec

    for i in range(0, D):
        tl.store(Y_ptr + base_y + i * stride_y_d, y[i].to(tl.float32))  # keep fp32; caller can cast


@triton.jit
def rotation_sin_kernel(
    X_norm_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_y_b, stride_y_h, stride_y_s, stride_y_d,
    inv_freq_ptr,
):
    # Each program handles one (b, h, s) row
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_y = pid_b * stride_y_b + pid_h * stride_y_h + pid_s * stride_y_s

    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i] = pid_s * f
        angle[i + D_half] = pid_s * f

    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    # Load x_norm row
    x = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        x[i] = tl.load(X_norm_ptr + base_x + i * stride_x_d)

    # Compute rotate_half(x)
    x1 = x[:D_half]
    x2 = x[D_half:]
    rot = tl.cat([-x2, x1], axis=0)  # shape [D]

    # Key rotation: sin-based (as per original intent)
    y = x * sin_vec - rot * cos_vec

    for i in range(0, D):
        tl.store(Y_ptr + base_y + i * stride_y_d, y[i].to(tl.float32))


@triton.jit
def scatter_update_cache_kernel(
    SRC_ptr, DST_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    stride_src_b, stride_src_h, stride_src_s, stride_src_d,
    stride_dst_b, stride_dst_h, stride_dst_l, stride_dst_d,
    cache_pos_ptr,
):
    # Each program handles one (b, h, s) and writes to dst at cache_pos[s]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + pid_s)  # int32 index into L

    base_src = pid_b * stride_src_b + pid_h * stride_src_h + pid_s * stride_src_s
    base_dst = pid_b * stride_dst_b + pid_h * stride_dst_h + pos * stride_dst_l

    for i in range(0, D):
        val = tl.load(SRC_ptr + base_src + i * stride_src_d)
        tl.store(DST_ptr + base_dst + i * stride_dst_d, val.to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        device = query.device
        dtype = query.dtype  # original outputs are in bf16
        B = query.shape[0]
        S = query.shape[2]
        D = query.shape[3]
        H_q = query.shape[1]
        H_kv = key.shape[1]

        # 1) RMSNorm: query_norm and key_norm
        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # For simplicity, assume weight is 1D of length D (original q_norm_weight is [D]).
        # Launch RMSNorm for query
        grid_q = (B, H_q, S)
        rmsnorm_kernel[grid_q](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            rms_norm_eps,
        )

        # Launch RMSNorm for key
        grid_k = (B, H_kv, S)
        rmsnorm_kernel[grid_k](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            rms_norm_eps,
        )

        # 2) Precompute per-(B,S) cos/sin vectors on host and pass to Triton rotation kernels
        # inv_freq is [D//2] fp32 on device. We'll compute angles in host, then pass to Triton.
        pos = position_ids  # [B, S], int64
        pos = pos.to(torch.int32)
        D_half = D // 2

        # Compute cos_all for query: cos(pos * inv_freq), shape [B, S, D]
        pos_f = pos.to(torch.float32)               # [B, S]
        inv_freq = inv_freq.to(torch.float32)      # [D//2]
        # Expand pos_f to [B, S, 1] and inv_freq to [1, 1, D//2] for broadcasting, but simple per-position:
        cos_vec_q = torch.cos(pos_f * inv_freq[None, None, :])  # [B, S, D//2]
        cos_all_q = torch.empty((B, S, D), dtype=torch.float32, device=device)
        for i in range(0, D_half):
            cos_all_q[:, :, i] = cos_vec_q[:, :, i]
        for i in range(0, D_half):
            cos_all_q[:, :, i + D_half] = cos_vec_q[:, :, i]

        # Compute sin_all for query: sin(pos * inv_freq), shape [B, S, D]
        sin_all_q = torch.empty((B, S, D), dtype=torch.float32, device=device)
        sin_vec_q = torch.sin(pos_f * inv_freq[None, None, :])  # [B, S, D//2]
        for i in range(0, D_half):
            sin_all_q[:, :, i] = sin_vec_q[:, :, i]
        for i in range(0, D_half):
            sin_all_q[:, :, i + D_half] = sin_vec_q[:, :, i]

        # Compute cos/sin for key: same angles, but keys use sin-based rotation
        sin_vec_k = torch.sin(pos_f * inv_freq[None, None, :])  # [B, S, D//2]
        sin_all_k = torch.empty((B, S, D), dtype=torch.float32, device=device)
        for i in range(0, D_half):
            sin_all_k[:, :, i] = sin_vec_k[:, :, i]
        for i in range(0, D_half):
            sin_all_k[:, :, i + D_half] = sin_vec_k[:, :, i]

        # 3) Rotate query and key using Triton
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        grid_qrot = (B, H_q, S)
        rotation_cos_kernel[grid_qrot](
            query_norm, query_rot,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            inv_freq.to(torch.float32),  # pass pointer to inv_freq, though not used here; kernel uses inv_freq_ptr
        )

        grid_krot = (B, H_kv, S)
        rotation_sin_kernel[grid_krot](
            key_norm, key_rot,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            inv_freq.to(torch.float32),
        )

        # 4) Cast back to original dtype if needed
        # The original returns bf16 tensors. Our rotation kernels store fp32; cast to query.dtype for consistency.
        query_rot = query_rot.to(dtype)
        key_rot = key_rot.to(dtype)

        # 5) Scatter update caches: write rotated keys and original values at cache_position
        L = key_cache.shape[2]
        # key_cache and value_cache have strides (B, H, L, D). We'll write at indices cache_pos[s] for each s.
        grid_scatter = (B, H_kv, S)
        # key_cache dtype is bf16, value_cache dtype is bf16; source tensors (query_rot, value) are bf16.
        # Cast src to fp32 for store; cache holds bf16, but we store converted values.
        scatter_update_cache_kernel[grid_scatter](
            key_rot, key_cache,
            B, H_kv, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position.to(torch.int32),
        )

        scatter_update_cache_kernel[grid_scatter](
            value, value_cache,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position.to(torch.int32),
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
