import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *const, [B, H, S, D] bf16
    y_ptr,          # *mut,   [B, H, S, D] bf16
    weight_ptr,     # *const, [D] bf16
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps,             # float32
    BLOCK_SIZE: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # 1) compute sum of squares across D in chunks
    sumsq = 0.0
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # 2) normalize and scale by weight, store
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = (x_fp32 * inv_rms) * w
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_q_kernel(
    x_ptr,          # [B, H_q, S, D] bf16 (RMSNormed)
    y_ptr,          # [B, H_q, S, D] bf16
    cos_ptr,        # [B, S, D] float32 (cos_all for query)
    sin_ptr,        # [B, S, D] float32 (sin_all for query)
    B, H_q, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load sin/cos vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)

    # Process columns in chunks
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        # rotate_half(x) = [-x2, x2, -x1, x1]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        y = x * sin - rotate_half * cos
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_k_kernel(
    x_ptr,          # [B, H_kv, S, D] bf16 (RMSNormed)
    y_ptr,          # [B, H_kv, S, D] bf16
    sin_ptr,        # [B, S, D] float32 (sin(angle), used as "sin_all" for keys)
    cos_ptr,        # [B, S, D] float32 (cos(angle), used as "cos_all" for keys)
    B, H_kv, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load sin/cos vectors for this (b, s)
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        sin = tl.load(sin_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=0.0).to(tl.float32)  # key uses sin(angle)
        cosv = tl.load(cos_ptr + b * 0 + s * 0 + cols * 0, mask=mask, other=1.0).to(tl.float32)  # and cos(angle)

    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        y = x * sin - rotate_half * cosv
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def scatter_update_cache_kernel(
    src_ptr,        # [B, H, S, D] bf16
    dst_ptr,        # [B, H, L, D] bf16
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # [S] int32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s)
    # Load src row [D] and store to dst[b, h, pos, :]
    for off in tl.static_range(0, D, 128):
        cols = off + tl.arange(0, 128)
        mask = cols < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        # Store as bf16 (dst_ptr is bf16); Triton will cast on store.
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l + cols * dst_stride_d, x, mask=mask)


def _build_cos_sin(angle_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    angle_vec: [D//2] float32 on device, e.g., pos * inv_freq
    Returns:
      cos_all_q: [B, S, D] float32
      sin_all_q: [B, S, D] float32
      sin_all_k: [B, S, D] float32 (keys use sin(angle))
      cos_all_k: [B, S, D] float32 (keys use cos(angle))
    """
    B = 1  # placeholder, will be set when launching kernels; these are not used in forward except precompute
    S = 1  # placeholders
    D = angle_vec.numel() * 2
    cos_half = torch.cos(angle_vec)             # [D//2]
    sin_half = torch.sin(angle_vec)             # [D//2]
    # Concatenate to full D for query rotation
    cos_all_q = torch.cat([cos_half, cos_half], dim=0)  # [D]
    sin_all_q = torch.cat([sin_half, sin_half], dim=0)  # [D]
    # For keys, original code uses sin(angle) rotation; we use sin as "cos_all" and cos as "sin_all"
    sin_all_k = cos_half.clone()  # placeholder, not used in rotation but preallocated
    cos_all_k = sin_half.clone()
    # Expand to [B, S, D] for kernels. In forward, we won't use these expansions; we pass cos/sin directly to Triton.
    # Return as device tensors with float32 dtype.
    return cos_all_q.float().to('cuda'), sin_all_q.float().to('cuda'), sin_all_k.float().to('cuda'), cos_all_k.float().to('cuda')


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
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
        """
        query: [B, num_q_heads, S, D], bf16
        key: [B, num_kv_heads, S, D], bf16
        value: [B, num_kv_heads, S, D], bf16
        position_ids: [B, S], int64
        key_cache: [B, num_kv_heads, L, D], bf16
        value_cache: [B, num_kv_heads, L, D], bf16
        cache_position: [S], int64
        q_norm_weight, k_norm_weight: [D], bf16
        inv_freq: [D//2], float32
        rms_norm_eps: float
        """
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16
        assert key_cache.dtype == torch.bfloat16 and value_cache.dtype == torch.bfloat16
        assert q_norm_weight.dtype == torch.bfloat16 and k_norm_weight.dtype == torch.bfloat16
        assert inv_freq.dtype == torch.float32

        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        H_kv = key.shape[1]
        L = key_cache.shape[2]

        # Ensure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        position_ids = position_ids.contiguous()
        cache_position = cache_position.contiguous()

        # Prepare position angles: angle[b, s, :] = pos * inv_freq, length D//2
        # Note: inv_freq is [D//2]. We build angle per (b, s).
        pos = position_ids  # [B, S], int64
        # Create angle vectors: [B, S, D//2]
        pos_flat = pos.view(B * S)  # [B*S]
        angle = pos_flat.unsqueeze(-1) * inv_freq.view(1, -1)  # [B*S, D//2], float32

        # Precompute cos_all_q, sin_all_q, sin_all_k, cos_all_k on device (float32)
        # We'll not use cat inside Triton; pass these vectors directly.
        # The following are device tensors with dtype float32, shapes [D] each.
        cos_all_q, sin_all_q, sin_all_k, cos_all_k = _build_cos_sin(angle.view(-1, D // 2)[0])  # dummy; not used

        # 1) RMSNorm for query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid = (B, H_q, S)
        rmsnorm_kernel[grid](
            query, query_norm, q_norm_weight, B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
        )

        grid = (B, H_kv, S)
        rmsnorm_kernel[grid](
            key, key_norm, k_norm_weight, B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
        )

        # 2) Query rotation: y = x * cos - rotate_half(x) * sin
        query_rot = torch.empty_like(query_norm)
        rotate_q_kernel[grid](
            query_norm, query_rot,
            # cos/sin tensors: we pass precomputed cos_all_q, sin_all_q as [B, S, D]
            # Note: Triton will interpret 0 strides as no-op; we pass correct tensors by shaping in Python.
            # Here, we pass sin_all_q as sin tensor and cos_all_q as cos tensor.
            sin_all_q, cos_all_q,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            BLOCK_SIZE=128,
        )

        # 3) Key rotation: y = x * sin(angle) - rotate_half(x) * cos(angle)
        key_rot = torch.empty_like(key_norm)
        rotate_k_kernel[grid](
            key_norm, key_rot,
            sin_all_k, cos_all_k,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_SIZE=128,
        )

        # 4) Scatter update caches
        # key_cache update: dst[b, h, cache_position[s], :] = query_rot[b, h, s, :]
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            query_rot, key_cache,
            B, H_kv, S, D, L,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position.to(torch.int32),
            BLOCK_SIZE=128,  # loop over D
        )

        # value_cache update: dst[b, h, cache_position[s], :] = value[b, h, s, :]
        # Note: original code sets value_cache to value (not rotated). We can use scatter_update_cache_kernel here too.
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            value, value_cache,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position.to(torch.int32),
            BLOCK_SIZE=128,
        )

        # Return tensors as in original: query_rotated, key_rotated, updated key_cache, updated value_cache
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
