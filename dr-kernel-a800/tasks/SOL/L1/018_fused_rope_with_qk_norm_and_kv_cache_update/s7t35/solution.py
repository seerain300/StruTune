import torch
import triton
import triton.language as tl


# -------- RMSNorm kernel: y = x * (weight * sqrt(D) / sqrt(mean(x^2) + eps)) --------

@triton.jit
def rmsnorm_kernel(
    x_ptr,       # *const T, input
    w_ptr,       # *const T, weight vector [D] (same dtype as x)
    y_ptr,       # *T, output
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    # Base pointers for this (b, h, s) row
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Accumulate sum of squares across D in fp32
    sum_x2 = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)
        sum_x2 += tl.sum(x_vals_f32 * x_vals_f32, axis=0)

    mean_x2 = sum_x2 / D
    inv_rms = tl.rsqrt(mean_x2 + eps)
    scale = tl.sqrt(D) * inv_rms  # scalar

    # Apply weight scaling
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals.to(tl.float32) * (scale * w_vals)
        # Store back in original dtype (bf16), Triton will cast on store
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


# -------- Compute cos_all and sin_all in Triton --------

@triton.jit
def compute_cos_kernel(
    pos_ptr,      # *const int32, shape [B, S] (we pass 2D pointer; we can index as pos[b, s])
    inv_ptr,      # *const float32, shape [D//2]
    cos_ptr,      # *float32, output [B, S, D]
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    # Load position
    pos = tl.load(pos_ptr + b * S + s)  # assuming pos_ids is [B, S] with contiguous rows
    # Compute angles: angle[:D_HALF] = pos * inv[:D_HALF]
    d_half = tl.arange(0, D_HALF)
    angles = pos.to(tl.float32) * tl.load(inv_ptr + d_half)  # [D_HALF]
    cos_vals = tl.cos(angles)  # [D_HALF]
    # Concatenate cos with itself to get [D]
    d = tl.arange(0, D)
    # d < D_HALF: cos_vals[d], else: cos_vals[d - D_HALF]
    cos_all = tl.where(d < D_HALF, cos_vals, cos_vals + D_HALF)  # placeholder logic
    # Simpler: create cos_all as [D] via broadcasting by filling: cos_all[d] = cos_vals[d % D_HALF]
    # Triton does not support Python-side list indexing; implement via selection:
    cos_all = tl.zeros([D], dtype=tl.float32)
    cos_all[:D_HALF] = cos_vals
    cos_all[D_HALF:] = cos_vals
    # Store to cos_ptr[b, s, :]
    out_ptr = cos_ptr + b * (S * D) + s * D
    tl.store(out_ptr + d, cos_all)


@triton.jit
def compute_sin_kernel(
    pos_ptr,      # *const int32, shape [B, S]
    inv_ptr,      # *const float32, shape [D//2]
    sin_ptr,      # *float32, output [B, S, D]
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr + b * S + s)
    d_half = tl.arange(0, D_HALF)
    angles = pos.to(tl.float32) * tl.load(inv_ptr + d_half)  # [D_HALF]
    sin_vals = tl.sin(angles)  # [D_HALF]
    sin_all = tl.zeros([D], dtype=tl.float32)
    sin_all[:D_HALF] = sin_vals
    sin_all[D_HALF:] = sin_vals
    out_ptr = sin_ptr + b * (S * D) + s * D
    tl.store(out_ptr + d, sin_all)


# -------- Rotate kernels --------

@triton.jit
def rotate_query_kernel(
    x_ptr,       # *const T (normalized), [B, H, S, D]
    cos_ptr,     # *const float32, [B, S, D]
    sin_ptr,     # *const float32, [B, S, D]
    y_ptr,       # *T, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    cos_stride0, cos_stride1, cos_stride2,  # cos/sin are [B, S, D], we pass strides
    sin_stride0, sin_stride1, sin_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Load cos_all and sin_all for this (b, s)
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    d = tl.arange(0, D)
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32

    # Loop over D in blocks, compute rotated output
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        # First half and second half
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first  # rotate_half(x) = [-x2, x1]
        y_vals = x_vals * cos_all - rotate * sin_all
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


@triton.jit
def rotate_key_kernel(
    x_ptr,       # *const T (normalized), [B, H, S, D]
    sin_ptr,     # *const float32, [B, S, D]
    cos_ptr,     # *const float32, [B, S, D]
    y_ptr,       # *T, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    sin_stride0, sin_stride1, sin_stride2,
    cos_stride0, cos_stride1, cos_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    d = tl.arange(0, D)
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first
        y_vals = x_vals * sin_all - rotate * cos_all
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


# -------- Scatter update kernels (write rotated key/value into cache at cache_position) --------

@triton.jit
def scatter_update_key_kernel(
    src_ptr,     # *const T (rotated key), [B, H, S, D]
    dst_ptr,     # *T (key_cache), [B, H, L, D]
    pos_ptr,     # *const int32, [S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(pos_ptr + s).to(tl.int32)
    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2
    dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * src_stride3)
        tl.store(dst_row_ptr + d * dst_stride3, val)


@triton.jit
def scatter_update_value_kernel(
    src_ptr,     # *const T (value), [B, H, S, D]
    dst_ptr,     # *T (value_cache), [B, H, L, D]
    pos_ptr,     # *const int32, [S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(pos_ptr + s).to(tl.int32)
    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2
    dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * src_stride3)
        tl.store(dst_row_ptr + d * dst_stride3, val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_q_heads: int = 96, num_kv_heads: int = 8, head_dim: int = 128, cache_len: int = 0):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_len = cache_len
        self.eps = 1e-6

    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        assert len(args) == 11, "Expected 11 inputs"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, _ = args

        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert H_q == self.num_q_heads, f"num_q_heads mismatch: expected {self.num_q_heads}, got {H_q}"
        H_kv = key.shape[1]
        assert H_kv == self.num_kv_heads, f"num_kv_heads mismatch: expected {self.num_kv_heads}, got {H_kv}"
        assert D == self.head_dim, f"head_dim mismatch: expected {self.head_dim}, got {D}"

        # Ensure device consistency
        device = query.device
        position_ids = position_ids.to(device)
        cache_position = cache_position.to(device)
        q_norm_weight = q_norm_weight.to(device)
        k_norm_weight = k_norm_weight.to(device)

        # Normalize query and key using Triton
        normalized_query = torch.empty_like(query)
        normalized_key = torch.empty_like(key)

        xq_grid = (B, H_q, S)
        rmsnorm_kernel[xq_grid](
            query, q_norm_weight, normalized_query,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            normalized_query.stride(0), normalized_query.stride(1), normalized_query.stride(2), normalized_query.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        xk_grid = (B, H_kv, S)
        rmsnorm_kernel[xk_grid](
            key, k_norm_weight, normalized_key,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            normalized_key.stride(0), normalized_key.stride(1), normalized_key.stride(2), normalized_key.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # Compute cos_all and sin_all in Triton: [B, S, D] float32
        # Note: Triton kernels expect indices as simple linear addressing; pass position_ids as 1D int32 [B*S].
        pos1d = position_ids.reshape(B * S).to(torch.int32)
        D_half = D // 2
        cos_all = torch.empty((B, S, D), dtype=torch.float32, device=device)
        sin_all = torch.empty((B, S, D), dtype=torch.float32, device=device)

        inv = inv_freq.to(device).to(torch.float32)

        compute_cos_kernel[(B, S)](
            pos1d, inv, cos_all,
            B, S, D, D_half,
        )

        compute_sin_kernel[(B, S)](
            pos1d, inv, sin_all,
            B, S, D, D_half,
        )

        # Rotate query and key using Triton
        query_rot = torch.empty_like(query)
        key_rot = torch.empty_like(key)

        rotate_q_grid = (B, H_q, S)
        rotate_query_kernel[rotate_q_grid](
            normalized_query, cos_all, sin_all, query_rot,
            B, H_q, S, D,
            normalized_query.stride(0), normalized_query.stride(1), normalized_query.stride(2), normalized_query.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            BLOCK_SIZE=128,
        )

        rotate_k_grid = (B, H_kv, S)
        rotate_key_kernel[rotate_k_grid](
            normalized_key, sin_all, cos_all, key_rot,
            B, H_kv, S, D,
            normalized_key.stride(0), normalized_key.stride(1), normalized_key.stride(2), normalized_key.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            BLOCK_SIZE=128,
        )

        # Update caches using Triton scatter
        # Cast key_rot and value to bf16 for store into caches
        key_rot_bf = key_rot.to(torch.bfloat16)
        value_bf = value.to(torch.bfloat16)

        L = key_cache.shape[2]
        # Ensure cache writes are correct: per (b, h, s), write to cache_position[s]
        # grid = (B, H_kv, S)
        scatter_k_grid = (B, H_kv, S)
        scatter_update_key_kernel[scatter_k_grid](
            key_rot_bf, key_cache, cache_position,
            B, H_kv, S, D, L,
            key_rot_bf.stride(0), key_rot_bf.stride(1), key_rot_bf.stride(2), key_rot_bf.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        )

        scatter_v_grid = (B, H_kv, S)
        scatter_update_value_kernel[scatter_v_grid](
            value_bf, value_cache, cache_position,
            B, H_kv, S, D, L,
            value_bf.stride(0), value_bf.stride(1), value_bf.stride(2), value_bf.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        )

        # Return query_rotated, key_rotated, updated key_cache, updated value_cache
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
