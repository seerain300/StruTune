import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr,
                         stride_xb, stride_xd, stride_yb, stride_yd,
                         M, D, eps):
    """
    RMSNorm over last dimension of shape (M, D). Each program handles one row.
    X_ptr, Y_ptr are pointers to [M, D] arrays with given strides. eps: float32.
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return
    sum_sq = 0.0
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * stride_xb + d * stride_xd)
        sum_sq += x.to(tl.float32) * x.to(tl.float32)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * stride_xb + d * stride_xd)
        y = x.to(tl.float32) / r
        tl.store(Y_ptr + row_id * stride_yb + d * stride_yd, y)


@triton.jit
def rotate_and_scale_kernel(X_ptr, Y_ptr,
                            B, S, D,
                            inv_ptr, pos_ptr,
                            stride_xb, stride_xs, stride_xd,
                            stride_yb, stride_ys, stride_yd):
    """
    Rotate and scale one row per (b, s). Grid: (B, S).
    inv_ptr: [D] float32, position_ids: [B, S] int64, pos_ptr selects pos = position_ids[b, s].
    X_ptr: [B, S, D], Y_ptr: [B, S, D].
    """
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if b >= B or s >= S:
        return

    pos = tl.load(pos_ptr + b * S + s)
    # Build inv of length D by repeating [inv, inv] from inv_ptr (length D)
    inv = tl.load(inv_ptr + tl.arange(0, D))
    emb = pos.to(tl.float32) * inv  # [D]
    cosv = tl.cos(emb)  # [D]
    sinv = tl.sin(emb)  # [D]

    half = D // 2
    # Load halves of the row
    x1 = tl.load(X_ptr + b * stride_xb + s * stride_xs + tl.arange(0, half) * stride_xd)
    x2 = tl.load(X_ptr + b * stride_xb + s * stride_xs + (half + tl.arange(0, half)) * stride_xd)

    # rotate_half(x) = [-x2, x1]
    rotated2 = -x2
    rotated1 = x1

    # y = x1 * cos + rotate_half(x) * sin
    # cos/sin applied to both halves but using appropriate segments:
    y1 = x1.to(tl.float32) * cosv.to(tl.float32) + rotated2.to(tl.float32) * sinv[half:].to(tl.float32)
    y2 = rotated1.to(tl.float32) * sinv[:half].to(tl.float32)

    # Store back to Y
    tl.store(Y_ptr + b * stride_yb + s * stride_ys + tl.arange(0, half) * stride_yd, y1)
    tl.store(Y_ptr + b * stride_yb + s * stride_ys + (half + tl.arange(0, half)) * stride_yd, y2)


@triton.jit
def scatter_key_cache_kernel(KEY_ptr, YC_ptr,
                             B, N_HEADS, S, D,
                             pos_ptr,  # int64, length S
                             stride_k_b, stride_k_n, stride_k_s, stride_k_d,
                             stride_c_b, stride_c_n, stride_c_s, stride_c_d):
    """
    Scatter rotated key into key_cache at cache_position[s]. Grid: (B, N_HEADS).
    For each s in 0..S-1: key_cache[b, n, cache_position[s], :] = KEY[b, n, s, :]
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    if b >= B or n >= N_HEADS:
        return
    for s in range(0, S):
        dest_pos = tl.load(pos_ptr + s)
        src_ptrs = KEY_ptr + b * stride_k_b + n * stride_k_n + s * stride_k_s + tl.arange(0, D) * stride_k_d
        dest_ptrs = YC_ptr + b * stride_c_b + n * stride_c_n + dest_pos * stride_c_s + tl.arange(0, D) * stride_c_d
        vals = tl.load(src_ptrs)  # load as-is (bf16)
        tl.store(dest_ptrs, vals)


@triton.jit
def scatter_value_cache_kernel(VAL_ptr, YC_ptr,
                                B, N_HEADS, S, D,
                                pos_ptr,  # int64, length S
                                stride_v_b, stride_v_n, stride_v_s, stride_v_d,
                                stride_c_b, stride_c_n, stride_c_s, stride_c_d):
    """
    Scatter value into value_cache at cache_position[s]. Grid: (B, N_HEADS).
    For each s in 0..S-1: value_cache[b, n, cache_position[s], :] = VAL[b, n, s, :]
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    if b >= B or n >= N_HEADS:
        return
    for s in range(0, S):
        dest_pos = tl.load(pos_ptr + s)
        src_ptrs = VAL_ptr + b * stride_v_b + n * stride_v_n + s * stride_v_s + tl.arange(0, D) * stride_v_d
        dest_ptrs = YC_ptr + b * stride_c_b + n * stride_c_n + dest_pos * stride_c_s + tl.arange(0, D) * stride_c_d
        vals = tl.load(src_ptrs)
        tl.store(dest_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, N_q, S, D = query.shape
        B_k, N_kv, S_k, D_k = key.shape
        assert B == B_k and S == S_k and D == D_k, "Incompatible shapes"

        # inv_freq is float32 length D//2; build inv of length D: [inv_freq, inv_freq]
        half = D // 2
        inv = torch.empty(D, dtype=torch.float32, device=query.device)
        inv[:half] = inv_freq
        inv[half:] = inv_freq

        # 1) RMSNorm for query and key using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        # For query
        M_q = B * N_q * S
        grid_q = (M_q,)
        rmsnorm_rows_kernel[grid_q](
            query, query_norm,
            query.stride(0), query.stride(3), query_norm.stride(0), query_norm.stride(3),
            M_q, D, rms_norm_eps,
            num_warps=4, num_stages=2
        )
        # For key
        M_k = B * N_kv * S
        grid_k = (M_k,)
        rmsnorm_rows_kernel[grid_k](
            key, key_norm,
            key.stride(0), key.stride(3), key_norm.stride(0), key_norm.stride(3),
            M_k, D, rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # 2) Rotate and scale query and key using Triton (rotate_and_scale_kernel)
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)
        # position_ids shape is (B, S); make it 1D for Triton
        pos_1d = position_ids.reshape(B * S).contiguous()  # [B*S]
        grid_rot = (B, S)
        # Rotate query
        rotate_and_scale_kernel[grid_rot](
            query_norm, query_rot,
            B, S, D,
            inv, pos_1d,
            query_norm.stride(0), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(2), query_rot.stride(3),
            num_warps=4, num_stages=2
        )
        # Rotate key
        rotate_and_scale_kernel[grid_rot](
            key_norm, key_rot,
            B, S, D,
            inv, pos_1d,
            key_norm.stride(0), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(2), key_rot.stride(3),
            num_warps=4, num_stages=2
        )

        # 3) Scatter into key_cache and value_cache at cache_position using Triton
        grid_scatter = (B, N_kv)
        scatter_key_cache_kernel[grid_scatter](
            key_rot, key_cache,
            B, N_kv, S, D,
            cache_position,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            num_warps=4, num_stages=2
        )
        scatter_value_cache_kernel[grid_scatter](
            value, value_cache,
            B, N_kv, S, D,
            cache_position,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=4, num_stages=2
        )

        # Return rotated query, rotated key, updated caches
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
