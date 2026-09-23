import torch
import triton
import triton.language as tl


# RMSNorm kernel: normalize along last dim D, then multiply by weight.
# y = x / rms * weight, where rms = sqrt(mean(x^2) + eps), computed in fp32, stored in x.dtype.
@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *input* pointer
    y_ptr,          # *output* pointer
    weight_ptr,     # *weight* pointer (length D)
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Accumulate sum of squares across D
    sumsq = 0.0
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * stride_b + h * stride_h + s * stride_s + cols * stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and scale by weight, store back
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * stride_b + h * stride_h + s * stride_s + cols * stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = (x_fp32 * inv_rms) * w
        # Store; Triton will cast to y_ptr dtype if needed
        tl.store(y_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s + cols * out_stride_d, y, mask=mask)


# Query rotation kernel: y = x * cos - rotate_half(x) * sin
@triton.jit
def rotate_q_kernel(
    x_ptr,          # *input* pointer (RMSNormed query), [B, H_q, S, D]
    y_ptr,          # *output* pointer, [B, H_q, S, D]
    cos_ptr,        # *cos* pointer, [B, S, D], bf16
    sin_ptr,        # *sin* pointer, [B, S, D], bf16
    B, H_q, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load cos/sin vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        cos = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)  # cos_ptr is [B, S, D], strides handled via b/s/cols
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)

    # Process columns
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        # rotate_half(x): [-x[D/2:], x[:D/2]]
        half = D // 2
        x_half = x[cols >= half]
        x_half = x[cols >= half] if half <= cols else x[:half]  # Note: Triton vector slicing must be done carefully; we recompute via mask
        # The above snippet needs careful correction: Triton doesn't support dynamic slicing like Python. Instead, load two halves explicitly:
        # Load first half
        x1 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + tl.minimum(cols, half - 1) * x_stride_d, mask=mask & (cols < half), other=0.0).to(tl.float32)
        # Load second half
        x2 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + (cols - half) * x_stride_d, mask=mask & (cols >= half), other=0.0).to(tl.float32)
        # Form rotated
        xr = -x2 * sin + x1 * cos  # incorrect due to mixing sin/cos; correct below

        # Correct formulation:
        # First compute x1 and x2 from y_ptr (we need x before rotation). Simpler: load x once, then construct rotated from x using masks.

        # Instead of trying to derive from x_ptr, better: just load x and compute rotate_half from x as:
        # We cannot index x with two halves; so we recompute using original x and masks.
        # Load x again for rotation computation
        x_loaded = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        # We need two vectors: first half and second half of x_loaded
        # Triton doesn't allow indexing with another vector, so we compute via two masked loads:
        # First half: cols < half
        x1 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask & (cols < half), other=0.0).to(tl.float32)
        # Second half: cols >= half
        x2 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + (cols - half) * x_stride_d, mask=mask & (cols >= half), other=0.0).to(tl.float32)
        xr = x1 * cos - (-x2) * sin  # using sin for key rotation (original code does key with sin)
        # Store back
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, xr, mask=mask)


# Key rotation kernel: y = x * sin - rotate_half(x) * cos
@triton.jit
def rotate_k_kernel(
    x_ptr,          # *input* pointer (RMSNormed key), [B, H_kv, S, D]
    y_ptr,          # *output* pointer, [B, H_kv, S, D]
    sin_ptr,        # *sin* pointer, [B, S, D], bf16
    cos_ptr,        # *cos* pointer, [B, S, D], bf16
    B, H_kv, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load cos/sin vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)
        cos = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)

    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        # First half and second half of x
        x1 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask & (cols < half), other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + (cols - half) * x_stride_d, mask=mask & (cols >= half), other=0.0).to(tl.float32)
        # rotate_half(x) = [-x2, x1]
        xr = x1 * sin - (-x2) * cos
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, xr, mask=mask)


# Cache scatter kernel: write rotated keys into key_cache at cache_position[s]
@triton.jit
def scatter_key_kernel(
    src_ptr,        # *rotated key* pointer, [B, H_kv, S, D]
    dst_ptr,        # *key_cache* pointer, [B, H_kv, L, D]
    cache_pos_ptr,  # *cache_position* pointer, [S] int32
    B, H_kv, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_s, dst_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    l = tl.load(cache_pos_ptr + s)  # int32

    # Loop over D in chunks
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + l * dst_stride_s + cols * dst_stride_d, x, mask=mask)


@triton.jit
def scatter_value_kernel(
    src_ptr,        # *value* pointer, [B, H_kv, S, D]
    dst_ptr,        # *value_cache* pointer, [B, H_kv, L, D]
    cache_pos_ptr,  # *cache_position* pointer, [S] int32
    B, H_kv, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_s, dst_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    l = tl.load(cache_pos_ptr + s)  # int32

    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + l * dst_stride_s + cols * dst_stride_d, x, mask=mask)


def run(
    query: torch.Tensor,            # [B, num_q_heads, S, D], bf16
    key: torch.Tensor,              # [B, num_kv_heads, S, D], bf16
    value: torch.Tensor,            # [B, num_kv_heads, S, D], bf16
    position_ids: torch.Tensor,     # [B, S], int64
    key_cache: torch.Tensor,        # [B, num_kv_heads, L, D], bf16
    value_cache: torch.Tensor,      # [B, num_kv_heads, L, D], bf16
    cache_position: torch.Tensor,   # [S], int64
    q_norm_weight: torch.Tensor,    # [D], bf16
    k_norm_weight: torch.Tensor,    # [D], bf16
    inv_freq: torch.Tensor,         # [D//2], float32
    rms_norm_eps: float,
):
    B, H_q, S, D = query.shape
    assert key.shape == (B, H_k := key.shape[1], S, D), "key shape mismatch"
    assert value.shape == (B, H_k, S, D), "value shape mismatch"
    num_kv_heads = H_k

    device = query.device
    dtype = query.dtype  # bf16

    # 1) RMSNorm on query and key
    # Allocate outputs
    query_norm = torch.empty_like(query)
    key_norm = torch.empty_like(key)

    # Launch RMSNorm kernel for query and key
    grid_q = (B, H_q, S)
    rmsnorm_kernel[grid_q](
        query, query_norm, q_norm_weight,
        B, H_q, S, D,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        rms_norm_eps,
        BLOCK_SIZE=128,
    )

    grid_k = (B, num_kv_heads, S)
    rmsnorm_kernel[grid_k](
        key, key_norm, k_norm_weight,
        B, num_kv_heads, S, D,
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        rms_norm_eps,
        BLOCK_SIZE=128,
    )

    # 2) Precompute cos_all/sin_all for query rotation (position_ids and inv_freq given)
    # Compute angle = pos * inv_freq (float32), then cos/sin, then concatenate with itself to form [B, S, D] bf16
    pos = position_ids.to(torch.float32)  # [B, S]
    angle = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(-1)  # [B, S, D/2] float32
    cos_base = angle.cos()  # [B, S, D/2]
    sin_base = angle.sin()  # [B, S, D/2]
    cos_all = torch.cat([cos_base, cos_base], dim=-1)  # [B, S, D] float32
    sin_all = torch.cat([sin_base, sin_base], dim=-1)  # [B, S, D] float32

    cos_all_bf16 = cos_all.to(torch.bfloat16)  # [B, S, D] bf16
    sin_all_bf16 = sin_all.to(torch.bfloat16)  # [B, S, D] bf16

    # 3) Rotate query: y = x * cos - rotate_half(x) * sin
    query_rot = torch.empty_like(query_norm)
    grid_qrot = (B, H_q, S)
    rotate_q_kernel[grid_qrot](
        query_norm, query_rot, cos_all_bf16, sin_all_bf16,
        B, H_q, S, D,
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
        BLOCK_SIZE=128,
    )

    # 4) Rotate key: y = x * sin - rotate_half(x) * cos (note: using sin for key)
    key_rot = torch.empty_like(key_norm)
    grid_krot = (B, num_kv_heads, S)
    rotate_k_kernel[grid_krot](
        key_norm, key_rot, sin_all_bf16, cos_all_bf16,
        B, num_kv_heads, S, D,
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        BLOCK_SIZE=128,
    )

    # 5) Cache scatter: write rotated keys and values into key/value caches at positions cache_position
    L = key_cache.shape[2]
    cp = cache_position.to(torch.int32)  # [S] int32

    # For key_cache
    grid_scatter_key = (B, num_kv_heads, S)
    scatter_key_kernel[grid_scatter_key](
        key_rot, key_cache,
        cp,
        B, num_kv_heads, S, D, L,
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        BLOCK_SIZE=128,
    )

    # For value_cache
    grid_scatter_val = (B, num_kv_heads, S)
    scatter_value_kernel[grid_scatter_val](
        value, value_cache,
        cp,
        B, num_kv_heads, S, D, L,
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        BLOCK_SIZE=128,
    )

    return query_rot, key_rot, key_cache, value_cache


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same 10 inputs as the original: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        # We repackage args into named variables for clarity.
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args
        return run(query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps)


def run(*args):
    return ModelNew()(*args)
