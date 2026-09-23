import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm across last dimension D for a tensor of shape [B, H, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr,          # *T, input tensor
    W_ptr,          # *T, weight tensor (same dtype as X)
    Y_ptr,          # *T, output tensor
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    eps: tl.float32,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    sumsq = 0.0
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        sumsq += x.to(tl.float32) * x.to(tl.float32)
    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(W_ptr + i).to(tl.float32)
    for i in range(0, D):
        x = tl.load(X_ptr + base + i * stride_d)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(Y_ptr + base + i * stride_d, y.to(x.dtype))


# Triton kernel: apply rotation to normalized tensors.
# For each (b, h, s), build cos/sin vectors using inv_freq, then:
# - query: y = x * cos - rotate_half(x) * sin
# - key:   y = x * sin - rotate_half(x) * cos
@triton.jit
def rotation_kernel(
    X_norm_ptr,     # *T, normalized x
    Y_ptr,          # *T, output rotated tensor
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_b, stride_h, stride_s, stride_d,
    inv_freq_ptr,   # *fp32, length D//2
    use_cos,        # 1 -> query rotation (use cos), 0 -> key rotation (use sin)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s

    # Build angles for D positions using inv_freq[:D//2]
    D_half = D // 2
    angle = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D_half):
        f = tl.load(inv_freq_ptr + i)  # fp32
        angle[i] = pid_s * f
        angle[i + D_half] = pid_s * f

    # Compute cos/sin
    cos_vec = tl.cos(angle)
    sin_vec = tl.sin(angle)

    if use_cos == 1:
        # query rotation: cos-based
        x = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            x[i] = tl.load(X_norm_ptr + base + i * stride_d)
        x1 = x[:D_half]
        x2 = x[D_half:]
        rot = tl.cat([-x2, x1], axis=0)
        y = x * cos_vec - rot * sin_vec
    else:
        # key rotation: sin-based (original uses sin for keys)
        x = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            x[i] = tl.load(X_norm_ptr + base + i * stride_d)
        x1 = x[:D_half]
        x2 = x[D_half:]
        rot = tl.cat([-x2, x1], axis=0)
        y = x * sin_vec - rot * cos_vec

    for i in range(0, D):
        tl.store(Y_ptr + base + i * stride_d, y[i].to(tl.float32).to(tl.bfloat16))


# Triton kernel: scatter update caches at positions given by cache_position
# Copies a row from Y_rot into CACHE at (b, h, cache_position[s], :)
# For value_cache, it copies the original 'value' row into the same position.
@triton.jit
def scatter_update_cache_kernel(
    SRC_ptr,        # *T, source row (rotated tensor or original value)
    CACHE_ptr,      # *T, destination cache tensor
    CACHE_STRIDE_L, # stride along L for cache (typically D, since contiguous along L for a given head)
    cache_pos_ptr,  # *int64, length S
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    base_src = pid_b * src_stride_b + pid_h * src_stride_h + pid_s * src_stride_s

    idx = tl.load(cache_pos_ptr + pid_s).to(tl.int32)
    for i in range(0, D):
        val = tl.load(SRC_ptr + base_src + i * src_stride_d)
        tl.store(CACHE_ptr + pid_b * (H * CACHE_STRIDE_L) + pid_h * CACHE_STRIDE_L + idx * CACHE_STRIDE_L + i, val.to(tl.float32).to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes: query [B, H_q=96, S, D=128], key [B, H_kv=8, S, D], value [B, H_kv, S, D]
        B, H_q, S, D = query.shape
        B2, H_kv, S2, D2 = key.shape
        assert B == B2 and S == S2 and D == D2, "Shape mismatch"
        assert H_q == 96 and H_kv == 8 and D == 128, "Unexpected num_heads or head_dim"

        # 1) RMSNorm for query and key in Triton
        query_norm = torch.empty_like(query, dtype=torch.bfloat16, device=query.device)
        key_norm = torch.empty_like(key, dtype=torch.bfloat16, device=key.device)

        stride_b = H_q * S * D
        stride_h = S * D
        stride_s = D
        stride_d = 1

        # Launch RMSNorm for query
        grid_q = (B, H_q, S)
        rmsnorm_kernel[grid_q](
            query, q_norm_weight, query_norm,
            B, H_q, S, D,
            stride_b, stride_h, stride_s, stride_d,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # Launch RMSNorm for key
        stride_b_k = B * H_kv * S * D
        stride_h_k = S * D
        stride_s_k = D
        stride_d_k = 1

        grid_k = (B, H_kv, S)
        rmsnorm_kernel[grid_k](
            key, k_norm_weight, key_norm,
            B, H_kv, S, D,
            stride_b_k, stride_h_k, stride_s_k, stride_d_k,
            rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # 2) Rotation in Triton: compute cos/sin inside kernel and apply rotation
        query_rotated = torch.empty_like(query_norm, dtype=torch.bfloat16, device=query.device)
        key_rotated = torch.empty_like(key_norm, dtype=torch.bfloat16, device=key.device)

        # query rotation uses cos
        grid_qr = (B, H_q, S)
        rotation_kernel[grid_qr](
            query_norm, query_rotated,
            B, H_q, S, D,
            stride_b, stride_h, stride_s, stride_d,
            inv_freq,  # length D//2
            1,  # use_cos = 1 for query
            num_warps=4, num_stages=2
        )

        # key rotation uses sin
        grid_kr = (B, H_kv, S)
        rotation_kernel[grid_kr](
            key_norm, key_rotated,
            B, H_kv, S, D,
            stride_b, stride_h, stride_s, stride_d,
            inv_freq,
            0,  # use_cos = 0 for key (sin)
            num_warps=4, num_stages=2
        )

        # 3) Scatter update caches in Triton: key_cache and value_cache
        # For key_cache: copy rotated key rows into cache at cache_position[s]
        # Strides for key_cache: treat as [B, H_kv, L=262144, D], contiguous along L and D
        # We index by (b, h, idx, d). Since idx varies per s, we'll build address dynamically.
        # The kernel expects CACHE_STRIDE_L as stride along L for a given (b, h). For contiguous layout [B, H, L, D], stride along L is D for a fixed (b, h).
        CACHE_STRIDE_L = D

        grid_sc = (B, H_kv, S)
        scatter_update_cache_kernel[grid_sc](
            key_rotated, key_cache,
            CACHE_STRIDE_L,
            cache_position,
            B, H_kv, S, D,
            stride_b, stride_h, stride_s, stride_d,
            num_warps=4, num_stages=2
        )

        # For value_cache: copy original value rows into cache at the same positions
        grid_sc_val = (B, H_kv, S)
        scatter_update_cache_kernel[grid_sc_val](
            value, value_cache,
            CACHE_STRIDE_L,
            cache_position,
            B, H_kv, S, D,
            stride_b, stride_h, stride_s, stride_d,
            num_warps=4, num_stages=2
        )

        # Return rotated query, rotated key, updated key_cache, updated value_cache
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
