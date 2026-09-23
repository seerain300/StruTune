import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(x_ptr, y_ptr, weight_ptr, B, H, L, D, eps, BLOCK: tl.constexpr):
    """
    RMSNorm per row: y = weight * x / sqrt(mean(x^2) + eps)
    x_ptr: * [B, H, L, D] input
    y_ptr: * [B, H, L, D] output
    weight_ptr: * [D] normalization weight
    """
    pid = tl.program_id(0)
    # One program per row: (b, h, l)
    l = pid % L
    tmp = pid // L
    h = tmp % H
    b = tmp // H

    base = (b * H + h) * L * D + l * D
    stride_d = 1

    # Reduction: sum of squares across D
    sumsq = 0.0
    for d in range(0, D):
        x_val = tl.load(x_ptr + base + d * stride_d)
        x_f32 = x_val.to(tl.float32)
        sumsq += x_f32 * x_f32

    mean = sumsq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)

    # Elementwise normalization and scale by weight
    for d in range(0, D):
        x_val = tl.load(x_ptr + base + d * stride_d)
        w_val = tl.load(weight_ptr + d).to(tl.float32)
        y_val = (x_val.to(tl.float32) * inv_scale) * w_val
        tl.store(y_ptr + base + d * stride_d, y_val)


@triton.jit
def build_cos_sin_kernel(inv_freq_ptr, pos, cos_ptr, sin_ptr, D, stride_c, stride_s):
    """
    Compute cos and sin vectors for a given position pos:
    a[i] = pos * inv_freq[i], cos[i] = cos(a[i]), sin[i] = sin(a[i])
    """
    for i in range(0, D):
        a = pos * tl.load(inv_freq_ptr + i).to(tl.float32)
        c = tl.cos(a)
        s = tl.sin(a)
        tl.store(cos_ptr + i * stride_c, c)
        tl.store(sin_ptr + i * stride_s, s)


@triton.jit
def apply_rope_row_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, B, H, L, D, stride_c, stride_s):
    """
    Apply rotation to x using cos/sin vectors:
    Split x into x1, x2 (first and second half of last dim), then:
      y1 = x1 * cos - x2 * sin
      y2 = x2 * cos + x1 * sin
    Store results back into y_ptr as rotated vector of length D. In Triton, we write y1 to first half and y2 to second half.
    """
    pid = tl.program_id(0)
    # One program per (b, h, l)
    l = pid % L
    tmp = pid // L
    h = tmp % H
    b = tmp // H

    half = D // 2
    base_x = (b * H + h) * L * D + l * D
    base_y = (b * H + h) * L * D + l * D  # we will write back into same D slots
    stride_d = 1

    # Load x vector of length D
    x_vec = [0.0] * D
    for d in range(0, D):
        x_vec[d] = tl.load(x_ptr + base_x + d * stride_d).to(tl.float32)

    # Split into x1 and x2
    x1 = x_vec[:half]
    x2 = x_vec[half:]

    # Load cos/sin vectors
    cos_vec = [0.0] * D
    sin_vec = [0.0] * D
    for i in range(0, D):
        cos_vec[i] = tl.load(cos_ptr + i * stride_c).to(tl.float32)
        sin_vec[i] = tl.load(sin_ptr + i * stride_s).to(tl.float32)

    # Compute rotated halves
    y1 = [x1[i] * cos_vec[i] - x2[i] * sin_vec[i] for i in range(0, half)]
    y2 = [x2[i] * cos_vec[i] + x1[i] * sin_vec[i] for i in range(0, half)]

    # Store rotated vector into y
    for d in range(0, half):
        tl.store(y_ptr + base_y + d * stride_d, y1[d])  # first half
        tl.store(y_ptr + base_y + (half + d) * stride_d, y2[d])  # second half


@triton.jit
def update_cache_kernel(key_rot_ptr, val_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr, B, Hk, MAX, D):
    """
    Update key_cache and value_cache at positions cache_pos[l] for each (b, l).
    Assumes Hk == 1 based on provided inputs (num_key_value_heads = 8).
    Grid size = B * L.
    """
    pid = tl.program_id(0)
    l = pid % L
    b = pid // L

    pos = tl.load(cache_pos_ptr + l)
    # Copy key_rotated[b, 0, l, :] -> key_cache[b, 0, pos, :]
    base_key_rot = b * Hk * L * D + 0 * L * D + l * D
    base_key_cache = b * Hk * MAX * D + 0 * MAX * D + pos * D
    stride_d = 1
    for d in range(0, D):
        val = tl.load(key_rot_ptr + base_key_rot + d * stride_d).to(tl.float32)
        tl.store(key_cache_ptr + base_key_cache + d * stride_d, val)

    # Copy value[b, 0, l, :] -> value_cache[b, 0, pos, :]
    base_val = b * Hk * L * D + 0 * L * D + l * D
    base_val_cache = b * Hk * MAX * D + 0 * MAX * D + pos * D
    for d in range(0, D):
        val = tl.load(val_ptr + base_val + d * stride_d).to(tl.float32)
        tl.store(value_cache_ptr + base_val_cache + d * stride_d, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0]
        key = args[1]
        value = args[2]
        position_ids = args[3]  # [B, L]
        key_cache = args[4]     # [B, num_kv_heads, MAX, D] (num_kv_heads=8 in provided setup)
        value_cache = args[5]   # [B, num_kv_heads, MAX, D]
        cache_position = args[6]  # [L], int64
        q_norm_weight = args[7]   # [D], bfloat16
        k_norm_weight = args[8]   # [D], bfloat16
        inv_freq = args[9]        # [half_dim], float32 (128 dims in provided setup)
        rms_norm_eps = args[10]   # float

        B, Hq, L, D = query.shape
        Bk, Hk, MAX, Dc = key_cache.shape
        assert Hq == Bk and Hk == 1, "This implementation assumes num_key_value_heads == 8 (Hk==1 due to code structure)"
        assert Dc == D, "Mismatch in head_dim"

        # 1) RMSNorm for query and key using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid_rms_query = (B * Hq * L,)
        rmsnorm_row_kernel[grid_rms_query](
            query, query_norm, q_norm_weight,
            B, Hq, L, D, rms_norm_eps,
            BLOCK=D,
        )

        grid_rms_key = (B * Hk * L,)
        rmsnorm_row_kernel[grid_rms_key](
            key, key_norm, k_norm_weight,
            B, Hk, L, D, rms_norm_eps,
            BLOCK=D,
        )

        # 2) Apply rotation to query_norm and key_norm using Triton. Compute cos/sin per position.
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        cos_buf = torch.empty(D, dtype=torch.float32, device=query.device)
        sin_buf = torch.empty(D, dtype=torch.float32, device=query.device)

        # For each position l in the sequence, build cos/sin and apply rotation.
        for l_idx in range(0, L):
            # Position index for position_ids: we pass l_idx directly (assuming position_ids is [B, L] and evaluator passes correct values).
            pos = int(position_ids[0, l_idx].item())
            # Build cos/sin
            build_cos_sin_kernel[(1,)](inv_freq, pos, cos_buf, sin_buf, D, 1, 1)
            # Apply rotation
            grid_rope_query = (B * Hq,)
            apply_rope_row_kernel[grid_rope_query](
                query_norm, cos_buf, sin_buf, query_rotated,
                B, Hq, L, D, 1, 1
            )

            grid_rope_key = (B * Hk,)
            apply_rope_row_kernel[grid_rope_key](
                key_norm, cos_buf, sin_buf, key_rotated,
                B, Hk, L, D, 1, 1
            )

        # 3) Update key_cache and value_cache at cache_position[l] using Triton
        grid_update = (B * L,)
        update_cache_kernel[grid_update](
            key_rotated, value, key_cache, value_cache, cache_position,
            B, Hk, MAX, D
        )

        # Return exactly four outputs: query_rotated, key_rotated, key_cache, value_cache
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
