import torch
import math
import triton
import triton.language as tl


# Triton kernel: RMSNorm over last dim D, per row (b, h, s).
# Input: x [B, H, S, D], weight [D] (1D), output y [B, H, S, D]
# We compute in fp32, store in original dtype (here bf16).
@triton.jit
def rmsnorm_row_kernel(
    X_ptr, Out_ptr, Weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # base pointer for this row
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # compute mean of x^2 across D
    mean_sum = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0)
        x = x.to(tl.float32)
        mean_sum += tl.sum(x * x, axis=0)

    mean = mean_sum / D
    inv_rms = tl.rsqrt(mean + eps)

    # scale and weight
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        # store back (Triton will cast to Out_ptr dtype if needed)
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: apply rotation for query:
# y = x * cos - rotate_half(x) * sin
# Input: x [B, H, S, D], cos_all [D], sin_all [D], output y [B, H, S, D]
@triton.jit
def rotate_query_kernel(
    X_ptr, Out_ptr, Cos_ptr, Sin_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # load cos/sin
    cos = tl.load(Cos_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < D, other=1.0).to(tl.float32)
    sin = tl.load(Sin_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < D, other=0.0).to(tl.float32)

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        half = x[:, :D // 2]
        other = x[:, D // 2:]
        rotated = -other * sin + half * cos  # shape (BLOCK_SIZE,)
        y = x * cos - rotated
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: apply rotation for key using sin_all only (original code uses sin for key):
# y = x * sin
# Here, sin_all must be provided; we ignore cos (original uses sin for key). If we need cos, we can derive it,
# but to keep exact semantics, we will use sin_all and ignore cos for key rotation.
@triton.jit
def rotate_key_sin_kernel(
    X_ptr, Out_ptr, Sin_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # load sin_all vector
    sin = tl.load(Sin_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < D, other=0.0).to(tl.float32)

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        y = x * sin
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: scatter update cache
# Inputs: src [B, H, S, D], Out [B, H, L, D], positions [B, S] int32
# For each (b, h, s): Out[b, h, positions[b, s], :] = src[b, h, s, :]
@triton.jit
def scatter_update_cache_kernel(
    Src_ptr, Out_ptr,
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    out_stride_b, out_stride_h, out_stride_l, out_stride_d,
    positions_ptr,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(positions_ptr + b * S + s)  # scalar int32
    src_row_ptr = Src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + pos * out_stride_l

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(src_row_ptr + idx * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row_ptr + idx * out_stride_d, x, mask=mask)


def _build_cos_sin_vectors(inv_freq, B, S, device):
    # inv_freq is [D//2] float32; we build [D] cos/sin vectors for each position.
    D = 2 * inv_freq.shape[0]
    pos_ids = torch.arange(S, dtype=torch.float32, device=device)  # [S]
    # Broadcast to (B, S, D//2)
    pos_ids = pos_ids[None, :, None].expand(B, S, D // 2)  # matches batch dimension by broadcasting B later
    # But better: build for one batch element and loop B:
    cos_all = []
    sin_all = []
    for i in range(B):
        angles = pos_ids * inv_freq  # [B, S, D//2]
        angles = angles.to(torch.float32)
        cos = torch.cos(angles)  # [B, S, D//2]
        sin = torch.sin(angles)  # [B, S, D//2]
        cos_full = torch.cat([cos, cos], dim=-1)  # [B, S, D]
        sin_full = torch.cat([sin, sin], dim=-1)  # [B, S, D]
        cos_all.append(cos_full)
        sin_all.append(sin_full)
    cos_all = torch.stack(cos_all, dim=0)  # [B, S, D]
    sin_all = torch.stack(sin_all, dim=0)  # [B, S, D]
    return cos_all.to(torch.bfloat16), sin_all.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters needed; we rely on provided inputs

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, H_q, S, D = query.shape
        B_kv, H_kv, S_k, D_k = key.shape
        assert B == B_kv and S == S_k and D == D_k, "query/key/value shapes must match"
        # Ensure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        # Build cos_all and sin_all using torch on device. This is allowed here (we don't call torch ops on tensors we must keep Triton-only).
        cos_all, sin_all = _build_cos_sin_vectors(inv_freq, B, S, query.device)

        # 1) RMSNorm on query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        grid_rms = (B, H_q, S)
        rmsnorm_row_kernel[grid_rms](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
        )
        grid_rms_key = (B, H_kv, S)
        rmsnorm_row_kernel[grid_rms_key](
            key, key_norm, k_norm_weight,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
        )

        # 2) Rotation
        # Query rotation: use cos_all
        query_rot = torch.empty_like(query_norm)
        grid_rot_q = (B, H_q, S)
        rotate_query_kernel[grid_rot_q](
            query_norm, query_rot, cos_all, sin_all,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_norm.stride(2), query_rot.stride(3),  # note: query_rot.stride(3) is correct here
            BLOCK_SIZE=128,
        )

        # Key rotation: use sin_all only (original code uses sin for key rotation)
        key_rot = torch.empty_like(key_norm)
        grid_rot_k = (B, H_kv, S)
        rotate_key_sin_kernel[grid_rot_k](
            key_norm, key_rot, sin_all,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_SIZE=128,
        )

        # 3) Cache update: scatter rotated keys and original values at cache_position
        # Convert cache_position to int32 for Triton
        cache_pos = cache_position.to(torch.int32)
        # Ensure key_cache, value_cache are bf16 as expected
        grid_scatter = (B, H_kv, S)
        scatter_update_cache_kernel[grid_scatter](
            key_rot, key_cache,
            B, H_kv, S, D, key_cache.shape[2],
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_pos,
            BLOCK_SIZE=128,
        )
        scatter_update_cache_kernel[grid_scatter](
            value, value_cache,
            B, H_kv, S, D, value_cache.shape[2],
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_pos,
            BLOCK_SIZE=128,
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
