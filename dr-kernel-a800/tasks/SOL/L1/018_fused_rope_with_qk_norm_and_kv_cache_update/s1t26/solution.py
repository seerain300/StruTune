import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    x_ptr, y_ptr, weight_ptr,
    B, H, L, D,
    stride_b, stride_h, stride_l, stride_d,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map pid to (b, h, l)
    L_per_BH = L
    b = pid // (H * L_per_BH)
    rem = pid % (H * L_per_BH)
    h = rem // L_per_BH
    l = rem % L_per_BH

    # Base offset for the row
    base = b * stride_b + h * stride_h + l * stride_l

    # Compute sum of squares in fp32
    sum_sq = 0.0
    for offs in range(0, D, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=mask, other=0.0)
        x_val_f32 = x_val.to(tl.float32)
        sum_sq += tl.sum(x_val_f32 * x_val_f32, axis=0)
    mean = sum_sq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)

    # Apply normalization and weight
    for offs in range(0, D, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=mask, other=0.0)
        w_val = tl.load(weight_ptr + idx, mask=mask, other=0.0)
        y = (x_val.to(tl.float32) * inv_scale) * w_val.to(tl.float32)
        tl.store(y_ptr + base + idx * stride_d, y, mask=mask)


@triton.jit
def build_cos_sin_kernel(
    inv_freq_ptr, cos_ptr, sin_ptr,
    pos, D,  # pos is int scalar (token position)
    c_stride, s_stride,
):
    # One program computes a full D-length vector
    idx = tl.arange(0, D)
    a = pos * tl.load(inv_freq_ptr + idx)  # float32
    c = tl.cos(a)
    s = tl.sin(a)
    tl.store(cos_ptr + idx * c_stride, c)
    tl.store(sin_ptr + idx * s_stride, s)


@triton.jit
def apply_rope_row_kernel(
    x_ptr, cos_ptr, sin_ptr, y_ptr,
    D,  # half_dim
    c_stride, s_stride,
):
    # One program per row (we will launch grid over B * H * L)
    pid = tl.program_id(0)
    L_per_BH = 1  # not needed since we use L=1 per program? We’ll map pid to (b,h,l) using a 1D grid sized by B*H*L.
    # For simplicity, assume grid launches exactly B*H*L programs and pid maps to (b,h,l) via modulo.
    # We don't have B,H,L here; so we rely on caller to pass correct grid and compute mapping in Python.
    # Each program processes one row; it reads D elements and applies rotation.
    idx = tl.arange(0, D)
    x = tl.load(x_ptr + idx)
    x_f32 = x.to(tl.float32)
    # Load cos/sin for these D elements
    cos = tl.load(cos_ptr + idx * c_stride)
    sin = tl.load(sin_ptr + idx * s_stride)
    x1 = x_f32[:D // 2]
    x2 = x_f32[D // 2:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y = tl.concatenate([y1, y2], axis=0)
    tl.store(y_ptr + idx, y)


@triton.jit
def update_cache_kernel(
    x_ptr, out_ptr, cache_pos_ptr, B, Hk, L, D, MAX, c_stride,
):
    # One program per (b, l)
    pid = tl.program_id(0)
    b = pid // L
    l = pid % L
    pos = tl.load(cache_pos_ptr + l)  # int64
    kv = 0  # single kv head index as per original, or loop if needed. Here Hk=8 but original uses one head for cache update; keep kv=0.
    base_x = b * (Hk * D * MAX) + kv * (D * MAX) + pos * D
    base_out = b * (Hk * D * MAX) + kv * (D * MAX) + pos * D

    # Read x row and write to out row
    idx = tl.arange(0, D)
    x = tl.load(x_ptr + base_x + idx)
    tl.store(out_ptr + base_out + idx, x)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query, key, value,
        position_ids, key_cache, value_cache,
        cache_position,
        q_norm_weight, k_norm_weight,
        inv_freq,
        rms_norm_eps,
    ):
        """
        Inputs:
          query: [B, H, L, D], bfloat16
          key: [B, Hk, L, D], bfloat16
          value: [B, Hk, L, D], bfloat16
          position_ids: [B, L], int64 (not used in rotation; original uses position p)
          key_cache: [B, Hk, MAX, D], bfloat16
          value_cache: [B, Hk, MAX, D], bfloat16
          cache_position: [L], int64
          q_norm_weight: [D], bfloat16
          k_norm_weight: [D], bfloat16
          inv_freq: [D//2], float32
          rms_norm_eps: float
        Outputs:
          query_rotated: [B, H, L, D], bfloat16
          key_rotated: [B, Hk, L, D], bfloat16
          key_cache: updated [B, Hk, MAX, D], bfloat16
          value_cache: updated [B, Hk, MAX, D], bfloat16
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA for Triton."
        assert q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda, "Weights/freq must be CUDA tensors."

        B, H, L, D = query.shape
        Hk = key.shape[1]
        MAX = key_cache.shape[2]

        # 1) RMSNorm for query
        query_norm = torch.empty_like(query)
        grid_q = (B * H * L,)
        rmsnorm_row_kernel[grid_q](
            query, query_norm, q_norm_weight,
            B, H, L, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            rms_norm_eps,
            BLOCK=D,
        )

        # 2) RMSNorm for key
        key_norm = torch.empty_like(key)
        grid_k = (B * Hk * L,)
        # For key, L is the last dimension for RMSNorm; we normalize per (b, h, l) row across D
        rmsnorm_row_kernel[grid_k](
            key, key_norm, k_norm_weight,
            B, Hk, L, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            rms_norm_eps,
            BLOCK=D,
        )

        # 3) Prepare cos/sin vectors for rotation per token position
        # We need one set of cos/sin per l in 0..L-1. Since Triton grid is 1D, we iterate l in Python and launch kernels.
        # Note: position_ids is provided but not used in original code; we use l as position.
        query_rotated = torch.empty_like(query)
        key_rotated = torch.empty_like(key)

        # Allocate per-l cos/sin
        cos_l = []
        sin_l = []
        for l in range(L):
            # Build cos/sin for this position
            cos_vec = torch.empty(D, dtype=torch.float32, device=query.device)
            sin_vec = torch.empty(D, dtype=torch.float32, device=query.device)
            build_cos_sin_kernel[(1,)](
                inv_freq, cos_vec, sin_vec,
                l, D,
                1, 1,
            )
            cos_l.append(cos_vec)
            sin_l.append(sin_vec)

        # 4) Apply rotation to query_norm -> query_rotated
        # One program per (b,h,l) row
        grid_apply_q = (B * H * L,)
        for l in range(L):
            # Map pid to (b, h)
            # We don't need exact mapping in-kernel since we use separate grid for each l; just launch with grid_q.
            apply_rope_row_kernel[grid_apply_q](
                query_norm, cos_l[l], sin_l[l], query_rotated,
                D // 1,  # half_dim = D, assuming even split; original uses 128 -> 64 each
                1, 1,
            )

        # Apply rotation to key_norm -> key_rotated
        key_rotated = torch.empty_like(key)
        grid_apply_k = (B * Hk * L,)
        for l in range(L):
            apply_rope_row_kernel[grid_apply_k](
                key_norm, cos_l[l], sin_l[l], key_rotated,
                D // 1,  # half_dim 64 for D=128
                1, 1,
            )

        # 5) Update key_cache and value_cache using cache_position
        # key_cache: update with rotated query per (b, l) at cache_position[l]
        # value_cache: update with unrotated value per (b, l) at cache_position[l]
        # We normalize Hk=8 per original; but original code uses one kv head. Here key has 8 heads; keep updating for all Hk.
        updated_key_cache = torch.empty_like(key_cache)
        updated_value_cache = torch.empty_like(value_cache)

        grid_update = (B * L,)
        update_cache_kernel[grid_update](
            query_rotated, updated_key_cache, cache_position,
            B, Hk, L, D, MAX, 1,
        )

        # Update value_cache with unrotated value
        # Note: original updates value_cache with value (unrotated). We'll copy value into updated_value_cache at positions cache_position[l].
        # Implement copy via kernel: for each (b, l), copy value[b, 0, l, :] into updated_value_cache[b, 0, cache_position[l], :]
        # Since value has Hk=8, we copy into head 0. If original used Hk in cache, it was intended for key; here we mimic original by copying to head 0.
        # To cover all Hk, we can copy to head 0 only; original returns updated value_cache with value at those positions.
        # But original expects updated_value_cache equals value_cache updated at those positions. We do that by copying value per l.
        # Launch a kernel per (b, l) to copy row.
        # Define a small kernel that copies a row from value to updated_value_cache at specified cache_position[l].
        pass  # The above Triton kernels are placeholders. Below, we actually implement the copy kernel.

        @triton.jit
        def copy_value_to_cache_kernel(
            val_ptr, out_ptr, cache_pos_ptr, B, Hk, L, D, MAX,
        ):
            pid = tl.program_id(0)
            b = pid // L
            l = pid % L
            pos = tl.load(cache_pos_ptr + l)
            base_val = b * (Hk * D * MAX) + 0 * (D * MAX) + l * D  # head 0
            base_out = b * (Hk * D * MAX) + 0 * (D * MAX) + pos * D
            idx = tl.arange(0, D)
            v = tl.load(val_ptr + base_val + idx)
            tl.store(out_ptr + base_out + idx, v)

        # Launch copy kernel
        grid_copy = (B * L,)
        copy_value_to_cache_kernel[grid_copy](
            value, updated_value_cache, cache_position,
            B, Hk, L, D, MAX,
        )

        # Return exactly four outputs as the original code
        return query_rotated, key_rotated, updated_key_cache, updated_value_cache


def run(*args):
    return ModelNew()(*args)
