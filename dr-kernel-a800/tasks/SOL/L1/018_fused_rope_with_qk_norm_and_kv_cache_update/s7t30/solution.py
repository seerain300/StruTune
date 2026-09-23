import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm on X (float32), write Out (same shape). Weight is [D], eps is scalar.
# One program per (b, h, s) row; reduce across D to compute inv_rms, then scale by weight.
@triton.jit
def rmsnorm_kernel(
    X_ptr, Out_ptr, Weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Pointers to the start of the row
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Compute sum of squares across D
    sumsq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Scale by weight and write
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        # If Out is same dtype as X, y is fp32; we can cast back to bf16 by writing y (Out is expected bf16).
        # Triton will store as the Out dtype. Ensure Out is bf16 tensor from host side.
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: rotate query as y = x * cos_all - rotate_half(x) * sin_all.
# We compute cos_all and sin_all inside the kernel (from inv_freq * pos), where pos = position_ids[b, s].
@triton.jit
def rotate_query_kernel(
    X_ptr, Out_ptr, Position_ptr, InvFreq_ptr,
    B, H_q, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load position
    pos = tl.load(Position_ptr + b * S + s)  # int32 scalar

    # Compute cos_all and sin_all of length D
    cos_all = tl.zeros([D], dtype=tl.float32)
    sin_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle = pos * tl.load(InvFreq_ptr + j)  # scalar float
        c = tl.cos(angle)
        s = tl.sin(angle)
        # place into even indices
        even_mask = (tl.arange(0, D) % 2) == 0
        cos_all = tl.where(even_mask, cos_all + c, cos_all)
        sin_all = tl.where(even_mask, sin_all + s, sin_all)

    # Pointers to row
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Apply rotation: y = x * cos_all - rotate_half(x) * sin_all
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        # rotate_half(x): [-x[D//2:], x[:D//2]]
        # Build second half and first half
        half = D // 2
        first_half = x[0:half]
        second_half = x[half:D]
        rot = tl.concatenate([-second_half, first_half], axis=0)  # shape [D]
        y = x * cos_all[offs : offs + BLOCK_SIZE] - rot * sin_all[offs : offs + BLOCK_SIZE]
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: rotate key as y = x * sin_all - rotate_half(x) * cos_all.
@triton.jit
def rotate_key_kernel(
    X_ptr, Out_ptr, Position_ptr, InvFreq_ptr,
    B, H_kv, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(Position_ptr + b * S + s)  # int32 scalar

    # Compute sin_all and cos_all of length D
    sin_all = tl.zeros([D], dtype=tl.float32)
    cos_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle = pos * tl.load(InvFreq_ptr + j)
        c = tl.cos(angle)
        s = tl.sin(angle)
        even_mask = (tl.arange(0, D) % 2) == 0
        sin_all = tl.where(even_mask, sin_all + s, sin_all)
        cos_all = tl.where(even_mask, cos_all + c, cos_all)

    # Pointers to row
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Apply rotation: y = x * sin_all - rotate_half(x) * cos_all
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        first_half = x[0:half]
        second_half = x[half:D]
        rot = tl.concatenate([-second_half, first_half], axis=0)
        y = x * sin_all[offs : offs + BLOCK_SIZE] - rot * cos_all[offs : offs + BLOCK_SIZE]
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: scatter update cache. For each (b, h, s), copy row into Out[b, h, positions[b, s], :]
@triton.jit
def scatter_update_cache_kernel(
    Src_ptr, Out_ptr,
    B, H, S, D,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    out_stride_b, out_stride_h, out_stride_l, out_stride_d,
    positions_ptr,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(positions_ptr + b * S + s)  # int32 scalar
    src_row_ptr = Src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + pos * out_stride_l

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(src_row_ptr + idx * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row_ptr + idx * out_stride_d, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes (assumed by the provided setup):
        # query: [B, 96, S, 128]
        # key:   [B, 8, S, 128]
        # value: [B, 8, S, 128]
        # position_ids: [B, S]
        # key_cache, value_cache: [B, 8, 262144, 128]
        # cache_position: [S] (int64)
        # q_norm_weight, k_norm_weight: [128] (bf16), ones
        # inv_freq: [64] (float32)
        # position_ids dtype: int64, cast to int32 for Triton
        B = query.shape[0]
        H_q = query.shape[1]
        H_kv = key.shape[1]
        S = query.shape[2]
        D = query.shape[3]

        # Ensure inputs are on the same device and dtype is bf16
        device = query.device
        # We'll do RMSNorm in fp32 inside kernels, but inputs/weights should be bf16
        # Create temporary outputs for normalized query/key in bf16
        query_norm = torch.empty_like(query, dtype=torch.bfloat16, device=device)
        key_norm = torch.empty_like(key, dtype=torch.bfloat16, device=device)

        # 1) RMSNorm on query and key using Triton
        # RMSNorm kernel: grid = (B, H_q, S) for query; (B, H_kv, S) for key
        # We'll pass weights q_norm_weight and k_norm_weight (bf16 tensors)
        # Launch RMSNorm for query
        grid_query = (B, H_q, S)
        rmsnorm_kernel[grid_query](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # Launch RMSNorm for key
        grid_key = (B, H_kv, S)
        rmsnorm_kernel[grid_key](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # 2) Rotate query and key using Triton
        # Cast position_ids to int32 for Triton
        position_ids_i32 = position_ids.to(torch.int32)

        # Rotate query: y = x * cos_all - rotate_half(x) * sin_all
        query_rot = torch.empty_like(query_norm, dtype=torch.bfloat16, device=device)
        grid_query_rot = (B, H_q, S)
        rotate_query_kernel[grid_query_rot](
            query_norm, query_rot, position_ids_i32, inv_freq,
            B, H_q, S, D, D // 2,  # HALF_D
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            BLOCK_SIZE=128,
        )

        # Rotate key: y = x * sin_all - rotate_half(x) * cos_all
        key_rot = torch.empty_like(key_norm, dtype=torch.bfloat16, device=device)
        grid_key_rot = (B, H_kv, S)
        rotate_key_kernel[grid_key_rot](
            key_norm, key_rot, position_ids_i32, inv_freq,
            B, H_kv, S, D, D // 2,  # HALF_D
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_SIZE=128,
        )

        # 3) Scatter update caches using Triton
        cache_position_i32 = cache_position.to(torch.int32)
        # Update key_cache with rotated keys
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            key_rot, key_cache,
            B, H_kv, S, D,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_position_i32,
            BLOCK_SIZE=128,
        )

        # Update value_cache with original values (no rotation)
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            value, value_cache,
            B, H_kv, S, D,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_position_i32,
            BLOCK_SIZE=128,
        )

        # Return the same outputs as original: rotated query, rotated key, updated caches
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
