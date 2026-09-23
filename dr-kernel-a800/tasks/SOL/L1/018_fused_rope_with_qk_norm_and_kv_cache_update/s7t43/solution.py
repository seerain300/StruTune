import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm along the last dimension (D) for a 4D tensor [B, H, S, D]
# Input x: fp16/bf16, weight: fp32 [D], output y: same dtype as x
@triton.jit
def rmsnorm_4d_kernel(
    x_ptr, y_ptr, weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Accumulate sum of squares across D in fp32
    sumsq = 0.0
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and apply weight, write back
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        w = tl.load(weight_ptr + d, mask=mask, other=1.0).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + d * y_stride_d, y, mask=mask)
        offs += BLOCK_SIZE

# Triton kernel: apply rotation to a 4D tensor [B, H, S, D] using provided cos_all/sin_all (length 2*D vectors)
# For query: y = x * cos - rotate_half(x) * sin
# For key: y = x * sin - rotate_half(x) * cos  (using sin_all as given in original logic)
@triton.jit
def rotate_4d_kernel(
    x_ptr, y_ptr, cos_ptr, sin_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Build half-dimension indices
    half = D // 2
    d = tl.arange(0, BLOCK_SIZE)
    mask = d < D

    # Load x
    x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0).to(tl.float32)

    # First half: d in [0, half)
    d0 = d  # 0..BLOCK_SIZE-1
    m0 = d0 < half
    x0 = x[d0]
    # Second half: d in [half, D)
    d1 = d0 + half
    m1 = d1 < D
    x1 = x[d1]

    # Load cos/sin for both halves
    cos0 = tl.load(cos_ptr + d0, mask=m0, other=0.0).to(tl.float32)
    cos1 = tl.load(cos_ptr + d1, mask=m1, other=0.0).to(tl.float32)
    sin0 = tl.load(sin_ptr + d0, mask=m0, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + d1, mask=m1, other=0.0).to(tl.float32)

    # Compute rotate: for query cos-all, sin-all; for key sin-all, cos-all per original code
    # We derive which to use from the fact that this kernel is launched twice: once with cos_ptr/cos_ptr, once with sin_ptr/sin_ptr.
    # Here we implement the general formula. The caller will pass appropriate cos_ptr/sin_ptr.
    # y = x * cos - (-rotate_half(x)) * sin for query
    # For key: y = x * sin - (-rotate_half(x)) * cos
    # Note: rotate_half(x) swaps halves: [-x1, x0]
    y0 = x0 * cos0 + x1 * sin1
    y1 = x1 * cos1 + x0 * sin0

    y = tl.where(d0 < half, y0, y1)

    # Store result
    tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + d * y_stride_d, y, mask=mask)

# Triton kernel: scatter update key/value caches at cache_position[s] for each batch and kv head
# y_src: [B, H_kv, S, D], y_dst: [B, H_kv, L, D]
@triton.jit
def scatter_update_cache_kernel(
    y_src_ptr, y_dst_ptr,
    B, H, S, D, L,
    y_src_stride_b, y_src_stride_h, y_src_stride_s, y_src_stride_d,
    y_dst_stride_b, y_dst_stride_h, y_dst_stride_s, y_dst_stride_d,
    cache_pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load cache position for this s
    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    # Copy y_src[b, h, s, :] to y_dst[b, h, pos, :]
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        src = tl.load(y_src_ptr + b * y_src_stride_b + h * y_src_stride_h + s * y_src_stride_s + d * y_src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_dst_ptr + b * y_dst_stride_b + h * y_dst_stride_h + pos * y_dst_stride_s + d * y_dst_stride_d, src, mask=mask)
        offs += BLOCK_SIZE

@triton.jit
def build_cos_sin_kernel(
    pos_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
    N, half_dim,
    pos_stride, inv_stride, cos_stride, sin_stride,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)  # 1D launch over N
    # Compute pos = pos_ptr[idx]
    pos = tl.load(pos_ptr + idx * pos_stride).to(tl.float32)
    # Build angle = pos * inv_freq[0:half_dim] in fp32
    k = 0
    while k < half_dim:
        d = k + tl.arange(0, BLOCK_SIZE)
        mask = d < half_dim
        inv = tl.load(inv_freq_ptr + d * inv_stride, mask=mask, other=0.0).to(tl.float32)
        angle = pos * inv
        c = tl.cos(angle)
        s = tl.sin(angle)
        tl.store(cos_ptr + idx * cos_stride + d, c, mask=mask)
        tl.store(sin_ptr + idx * sin_stride + d, s, mask=mask)
        k += BLOCK_SIZE

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ):
        # Ensure device and contiguity
        device = query.device
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be CUDA tensors"
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        B_q, H_q, S, D = query.shape
        Bk, H_kv, L, Dv = key_cache.shape
        assert H_q == 96, "This implementation expects num_q_heads=96."
        assert H_kv == 8, "This implementation expects num_kv_heads=8."
        assert D == 128 and Dv == 128, "This implementation expects head_dim=128."

        # 1) RMSNorm on query and key (compute in fp32, return in original dtype)
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid_rms = (B_q, H_q, S)
        rmsnorm_4d_kernel[grid_rms](
            query, query_norm, q_norm_weight.to(torch.float32),
            B_q, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            rms_norm_eps, BLOCK_SIZE=128, num_warps=4,
        )

        rmsnorm_4d_kernel[grid_rms](
            key, key_norm, k_norm_weight.to(torch.float32),
            B_q, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            rms_norm_eps, BLOCK_SIZE=128, num_warps=4,
        )

        # 2) Build rotation vectors cos_all and sin_all in Triton, per position s
        # Flatten position_ids to [B_q*S]
        N = B_q * S
        pos_vec = position_ids.reshape(N).to(torch.int32)
        half_dim = D // 2
        # Allocate per-position cos/sin vectors: shape [N, half_dim]
        cos_all = torch.empty((N, half_dim), dtype=torch.float32, device=device)
        sin_all = torch.empty((N, half_dim), dtype=torch.float32, device=device)

        # Launch kernel: 1D grid over N
        build_cos_sin_kernel[(N,)](
            pos_vec, inv_freq,
            cos_all, sin_all,
            N, half_dim,
            1, 1, 1, 1,
            BLOCK_SIZE=128, num_warps=4,
        )

        # Extend to full D for query: cos_all_q = cos_all cat with cos_all, sin_all_q likewise
        cos_all_q = torch.empty((N, D), dtype=torch.float32, device=device)
        sin_all_q = torch.empty((N, D), dtype=torch.float32, device=device)
        # First half
        cos_all_q[:, :half_dim] = cos_all
        sin_all_q[:, :half_dim] = sin_all
        # Second half: same vectors (original code repeats both cos and sin for query)
        cos_all_q[:, half_dim:] = cos_all
        sin_all_q[:, half_dim:] = sin_all

        # Extend to full D for key: sin_all_k = sin_all cat with sin_all, cos_all_k likewise
        cos_all_k = torch.empty((N, D), dtype=torch.float32, device=device)
        sin_all_k = torch.empty((N, D), dtype=torch.float32, device=device)
        # Key uses sin-based rotation per original logic: sin_all_k = sin_all repeated, cos_all_k = cos_all repeated
        cos_all_k[:, :half_dim] = cos_all
        cos_all_k[:, half_dim:] = cos_all
        sin_all_k[:, :half_dim] = sin_all
        sin_all_k[:, half_dim:] = sin_all

        # 3) Apply rotation in Triton
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        # Query rotation: use cos_all_q and sin_all_q
        grid_qrot = (B_q, H_q, S)
        rotate_4d_kernel[grid_qrot](
            query_norm, query_rot,
            cos_all_q, sin_all_q,
            B_q, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            BLOCK_SIZE=128, num_warps=4,
        )

        # Key rotation: use sin_all_k and cos_all_k
        grid_krot = (B_q, H_kv, S)
        rotate_4d_kernel[grid_krot](
            key_norm, key_rot,
            sin_all_k, cos_all_k,
            B_q, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_SIZE=128, num_warps=4,
        )

        # 4) Scatter update caches: write rotated keys and original values at cache_position
        # Ensure cache_position is int32 for Triton
        cache_pos_i32 = cache_position.to(torch.int32)

        # Scatter keys: key_rot shape [B_q, H_kv, S, D]
        grid_sc = (B_q, H_kv, S)
        scatter_update_cache_kernel[grid_sc](
            key_rot.to(torch.bfloat16), key_cache,
            B_q, H_kv, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_pos_i32,
            BLOCK_SIZE=128, num_warps=4,
        )

        # Scatter values: value shape [B_q, H_kv, S, D] -> store into value_cache [B_q, H_kv, L, D]
        # Note: original code writes original value (not rotated). Here we use 'value' tensor provided.
        grid_sc[grid_sc](  # reuse the same grid
            value.to(torch.bfloat16), value_cache,
            B_q, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_pos_i32,
            BLOCK_SIZE=128, num_warps=4,
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
