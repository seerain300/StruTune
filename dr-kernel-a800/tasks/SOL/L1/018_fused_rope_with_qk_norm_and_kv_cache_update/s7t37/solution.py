import torch
import triton
import triton.language as tl


# -------- RMSNorm kernel --------

@triton.jit
def rmsnorm_kernel(
    x_ptr,           # *const T_in (bf16), [B, H, S, D]
    w_ptr,           # *const T_w (bf16), [D]
    y_ptr,           # *T_out (bf16), [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    w_stride0,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Compute sum of squares across D
    sumsq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + 1e-6)  # eps from original code

    # Normalize and scale by weight
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptr + idx * w_stride0, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


# -------- Compute cos_all and sin_all for each (b, s) into [B, S, D] (fp32) --------

@triton.jit
def compute_cos_sin_kernel(
    pos_ptr,         # *const int32, [B, S]
    inv_freq_ptr,    # *const float32, [D//2]
    cos_ptr,         # *float32, [B, S, D]
    sin_ptr,         # *float32, [B, S, D]
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    cos_stride0, cos_stride1, cos_stride2,
    sin_stride0, sin_stride1, sin_stride2,
    pos_stride0, pos_stride1,
    inv_freq_stride0,
    HALF_D: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr + b * pos_stride0 + s * pos_stride1).to(tl.float32)
    angle = pos * tl.load(inv_freq_ptr + tl.arange(0, HALF_D) * inv_freq_stride0).to(tl.float32)  # [64]
    cos_vec = tl.cos(angle)  # [64]
    sin_vec = tl.sin(angle)  # [64]

    # Concatenate to length D: cos_all = [cos_vec, cos_vec], sin_all = [sin_vec, sin_vec]
    d = tl.arange(0, D)
    d_half = d < (D // 2)

    # For cos/sin, we need two halves mapping:
    # d in [0, 64): cos_vec[d]; d in [64, 128): cos_vec[d - 64]
    # d in [0, 64): sin_vec[d]; d in [64, 128): sin_vec[d - 64]
    # Build masks for first and second half
    mask_first_half = (d < (D // 2)) & (d < HALF_D)
    mask_second_half = (d >= (D // 2)) & (d < (D // 2) + HALF_D)

    # For cos, values:
    cos_all = tl.where(mask_first_half, cos_vec[d], tl.where(mask_second_half, cos_vec[d - (D // 2)], 0.0))
    sin_all = tl.where(mask_first_half, sin_vec[d], tl.where(mask_second_half, sin_vec[d - (D // 2)], 0.0))

    # Store to cos_ptr[b, s, :] and sin_ptr[b, s, :]
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    tl.store(cos_base + d * cos_stride2, cos_all)
    tl.store(sin_base + d * sin_stride2, sin_all)


# -------- Rotation kernels: per (b, h, s) --------

@triton.jit
def rotate_query_kernel(
    x_ptr,           # *const T (normalized), [B, H, S, D]
    cos_ptr,         # *const float32, [B, S, D]
    sin_ptr,         # *const float32, [B, S, D]
    y_ptr,           # *T_out (bf16), [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    cos_stride0, cos_stride1, cos_stride2,
    sin_stride0, sin_stride1, sin_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    d = tl.arange(0, D)
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride1 + s * sin_stride1  # sin_stride1 unused in body but keep signature
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first  # rotate_half(x) = [-x2, x1]
        y_vals = x_vals * cos_all - rotate * sin_all
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


@triton.jit
def rotate_key_kernel(
    x_ptr,           # *const T (normalized), [B, H, S, D]
    sin_ptr,         # *const float32, [B, S, D]
    cos_ptr,         # *const float32, [B, S, D]
    y_ptr,           # *T_out (bf16), [B, H, S, D]
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

    d = tl.arange(0, D)
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first  # rotate_half(x) = [-x2, x1]
        y_vals = x_vals * sin_all - rotate * cos_all
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


# -------- Scatter update kernels: write rotated keys and values to cache at cache_position[s] --------

@triton.jit
def scatter_update_kernel(
    src_ptr,         # *const T (bf16), [B, H, S, D]
    dst_ptr,         # *T (bf16), [B, H, L, D]
    pos_ptr,         # *const int32, [S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(pos_ptr + s).to(tl.int32)
    # Base pointers for src row [b, h, s, :]
    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2

    # Loop across D in blocks and store into dst[b, h, pos, :]
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        vals = tl.load(src_row_ptr + idx * src_stride3, mask=mask, other=0.0)  # load as bf16 directly
        dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2
        tl.store(dst_row_ptr + idx * dst_stride3, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_q_heads: int = 96, num_kv_heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = 1e-6

    def forward(self, *args):
        # Expect: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq
        assert len(args) == 10, "Expected 10 inputs: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq = args

        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert H_q == self.num_q_heads, f"num_q_heads mismatch: expected {self.num_q_heads}, got {H_q}"
        H_kv = key.shape[1]
        assert H_kv == self.num_kv_heads, f"num_kv_heads mismatch: expected {self.num_kv_heads}, got {H_kv}"
        assert D == self.head_dim, f"head_dim mismatch: expected {self.head_dim}, got {D}"
        assert key.shape == (B, H_kv, S, D) and value.shape == (B, H_kv, S, D)

        device = query.device

        # Ensure dtype compatibility: inv_freq should be fp32
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)

        # Prepare tensors for Triton kernels
        # 1) RMSNorm: outputs x_norm_query, x_norm_key
        x_norm_query = torch.empty_like(query)
        x_norm_key = torch.empty_like(key)

        # Launch RMSNorm kernels over all heads
        grid_rms = (B, H_q, S)
        rmsnorm_kernel[grid_rms](
            query, q_norm_weight.to(device=device, dtype=torch.bfloat16), x_norm_query,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            x_norm_query.stride(0), x_norm_query.stride(1), x_norm_query.stride(2), x_norm_query.stride(3),
            q_norm_weight.stride(0),
            BLOCK_SIZE=128,
        )
        grid_rms = (B, H_kv, S)
        rmsnorm_kernel[grid_rms](
            key, k_norm_weight.to(device=device, dtype=torch.bfloat16), x_norm_key,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            x_norm_key.stride(0), x_norm_key.stride(1), x_norm_key.stride(2), x_norm_key.stride(3),
            k_norm_weight.stride(0),
            BLOCK_SIZE=128,
        )

        # 2) Compute cos_all and sin_all using Triton: cos_ptr, sin_ptr shape [B, S, D], fp32
        pos = position_ids.to(device=device, dtype=torch.int32)
        B_pos = pos.shape[0]
        S_pos = pos.shape[1]
        assert B_pos == B and S_pos == S, "position_ids shape mismatch"
        cos_all = torch.empty((B, S, D), device=device, dtype=torch.float32)
        sin_all = torch.empty((B, S, D), device=device, dtype=torch.float32)
        half_d = D // 2

        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        # Ensure inv_freq is [D//2]
        inv_freq = inv_freq[:half_d]

        compute_cos_sin_kernel[(B, S)](
            pos, inv_freq,
            cos_all, sin_all,
            B, S, D,
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            pos.stride(0), pos.stride(1),
            inv_freq.stride(0),
            HALF_D=half_d,
        )

        # 3) Rotate query: y_query
        y_query = torch.empty_like(query)
        rotate_query_kernel[(B, H_q, S)](
            x_norm_query, cos_all, sin_all, y_query,
            B, H_q, S, D,
            x_norm_query.stride(0), x_norm_query.stride(1), x_norm_query.stride(2), x_norm_query.stride(3),
            y_query.stride(0), y_query.stride(1), y_query.stride(2), y_query.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            BLOCK_SIZE=128,
        )

        # 4) Rotate key: y_key (per kv head)
        y_key = torch.empty_like(key)
        rotate_key_kernel[(B, H_kv, S)](
            x_norm_key, sin_all, cos_all, y_key,
            B, H_kv, S, D,
            x_norm_key.stride(0), x_norm_key.stride(1), x_norm_key.stride(2), x_norm_key.stride(3),
            y_key.stride(0), y_key.stride(1), y_key.stride(2), y_key.stride(3),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            BLOCK_SIZE=128,
        )

        # 5) Scatter update caches: rotated keys and original values at cache_position[s]
        # key_cache: [B, H_kv, L, D], value_cache: [B, H_kv, L, D]
        L = key_cache.shape[2]
        assert value_cache.shape == (B, H_kv, L, D), "value_cache shape mismatch"
        # Ensure cache_position is int32
        cache_pos = cache_position.to(device=device, dtype=torch.int32)

        scatter_update_kernel[(B, H_kv, S)](
            y_key, key_cache,
            cache_pos,
            B, H_kv, S, D, L,
            y_key.stride(0), y_key.stride(1), y_key.stride(2), y_key.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        )
        scatter_update_kernel[(B, H_kv, S)](
            value, value_cache,
            cache_pos,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        )

        return y_query, y_key, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
