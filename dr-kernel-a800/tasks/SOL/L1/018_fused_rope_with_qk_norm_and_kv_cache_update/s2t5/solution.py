import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, stride_xm, stride_xd, stride_ym, stride_yd):
    """
    RMSNorm over last dimension of shape (M, D).
    Each program handles one row. Computes r = sqrt(mean(x^2) + eps) and writes y = x / r.
    X_ptr, Y_ptr are 2D pointers, but we treat as row-major with strides (stride_xm, stride_xd) and (stride_ym, stride_yd).
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return

    sum_sq = 0.0
    # Accumulate sum of squares over D in fp32
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * stride_xm + d * stride_xd)
        x_f32 = x.to(tl.float32)
        sum_sq += x_f32 * x_f32
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)

    # Scale and store
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * stride_xm + d * stride_xd)
        y = x / r
        tl.store(Y_ptr + row_id * stride_ym + d * stride_yd, y)


@triton.jit
def rotate_and_scale_kernel(X_ptr, Y_ptr, B, S, D, inv_ptr, pos_ids_ptr, stride_xb, stride_xn, stride_xs, stride_xd,
                            stride_yb, stride_yn, stride_ys, stride_yd):
    """
    For each (b, s), compute per-token cos/sin vectors of length D and apply rotation:
      x1 = X[b, :, s, :D//2], x2 = X[b, :, s, :D//2], rotate_half(x) = [-x2, x1]
      cos_vec = cos(pos * inv), sin_vec = sin(pos * inv), inv is length D//2 extended to D by repeating.
      Y = x1*cos_vec[:D//2] + rotate_half(x)*sin_vec[:D//2]
    Grid: (B, S)
    """
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (b >= B) or (s >= S):
        return

    pos = tl.load(pos_ids_ptr + b * S + s).to(tl.int32)
    half = D // 2

    # Build inv vector of length D: [inv[:half], inv[:half]]
    inv_half = tl.arange(0, half)
    inv_full = tl.arange(0, D)  # we'll fill first half with inv_ptr, second half with same
    cos_vec = tl.zeros((D,), dtype=tl.float32)
    sin_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, half):
        v = inv_ptr[i]
        c = tl.cos(pos.to(tl.float32) * v)
        s2 = tl.sin(pos.to(tl.float32) * v)
        cos_vec[i] = c
        sin_vec[i] = s2
    # second half repeats
    for i in range(0, half):
        cos_vec[i + half] = cos_vec[i]
        sin_vec[i + half] = sin_vec[i]

    # Load x row: for simplicity, we assume X_ptr is (B, N, S, D). We need to pick a specific head n? The original uses query and key of shape (B, N_q/S, S, D). Triton kernel needs to know N, but the forward will pass query and key separately; we’ll implement for general (B, N, S, D) tensors by decoding b and s and loading X[b, n, s, :].
    # However, Triton kernel cannot use unknown N here; we pass pointers for query and key separately in ModelNew.forward. For clarity, we implement loading for a fixed n, which we encode into grid axis or pass as separate pointers. To keep it general, we rely on ModelNew.forward to pass query/rotated pointers directly.

    # Note: Below we assume Y_ptr points to output tensor with same shape as X_ptr, and we will compute rotation per (b, s) for each head. In practice, we'll launch this kernel per (b, s) and pass appropriate pointers from forward. Triton grid covers B*S, and forward prepares inputs accordingly.

    # Placeholder: We need to load X and store to Y. Since we don't have N here, we can't implement the general load/store. We will instead launch this kernel with separate query and key pointers from ModelNew.forward and compute rotation accordingly. Triton doesn't support dynamic indexing of Python loops over N here; thus we provide two specialized kernels for query and key.

    # To satisfy the requirement, we will define specialized versions below and invoke from ModelNew.forward.


@triton.jit
def rotate_and_scale_query_kernel(Query_ptr, QueryR_ptr, B, NQ, S, D, inv_ptr, pos_ids_ptr,
                                  stride_qb, stride_qn, stride_qs, stride_qd,
                                  stride_rb, stride_rn, stride_rs, stride_rd):
    """
    Specialized kernel for query rotation: process all (b, n, s) and apply per-token cos/sin rotation.
    Grid: (B*NQ, S)
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    b = pid // NQ
    n = pid % NQ
    if (b >= B) or (s >= S):
        return
    pos = tl.load(pos_ids_ptr + b * S + s).to(tl.int32)
    half = D // 2

    # Build cos/sin vectors for token (b, s)
    inv_half = tl.arange(0, half)
    cos_vec = tl.zeros((D,), dtype=tl.float32)
    sin_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, half):
        v = inv_ptr[i]
        c = tl.cos(pos.to(tl.float32) * v)
        s2 = tl.sin(pos.to(tl.float32) * v)
        cos_vec[i] = c
        sin_vec[i] = s2
    for i in range(0, half):
        cos_vec[i + half] = cos_vec[i]
        sin_vec[i + half] = sin_vec[i]

    # Load original query row and compute rotated
    x = tl.load(Query_ptr + b * stride_qb + n * stride_qn + s * stride_qs + tl.arange(0, D) * stride_qd)
    x1 = x[:half]
    x2 = x[half:]
    rotated_half = -x2 * sin_vec[:half] + x1 * cos_vec[:half]
    y = tl.zeros((D,), dtype=x.dtype)
    y[:half] = rotated_half
    y[half:] = x1 * cos_vec[half:] + x2 * sin_vec[half:]

    tl.store(QueryR_ptr + b * stride_rb + n * stride_rn + s * stride_rs + tl.arange(0, D) * stride_rd, y)


@triton.jit
def rotate_and_scale_key_kernel(Key_ptr, KeyR_ptr, B, NKV, S, D, inv_ptr, pos_ids_ptr,
                                stride_kb, stride_kn, stride_ks, stride_kd,
                                stride_rkb, stride_rkn, stride_rks, stride_rkd):
    """
    Specialized kernel for key rotation: process all (b, n, s) and apply per-token cos/sin rotation.
    Grid: (B*NKV, S)
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    b = pid // NKV
    n = pid % NKV
    if (b >= B) or (s >= S):
        return
    pos = tl.load(pos_ids_ptr + b * S + s).to(tl.int32)
    half = D // 2

    # Build cos/sin vectors for token (b, s)
    inv_half = tl.arange(0, half)
    cos_vec = tl.zeros((D,), dtype=tl.float32)
    sin_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, half):
        v = inv_ptr[i]
        c = tl.cos(pos.to(tl.float32) * v)
        s2 = tl.sin(pos.to(tl.float32) * v)
        cos_vec[i] = c
        sin_vec[i] = s2
    for i in range(0, half):
        cos_vec[i + half] = cos_vec[i]
        sin_vec[i + half] = sin_vec[i]

    # Load original key row and compute rotated
    x = tl.load(Key_ptr + b * stride_kb + n * stride_kn + s * stride_ks + tl.arange(0, D) * stride_kd)
    x1 = x[:half]
    x2 = x[half:]
    rotated_half = -x2 * sin_vec[:half] + x1 * cos_vec[:half]
    y = tl.zeros((D,), dtype=x.dtype)
    y[:half] = rotated_half
    y[half:] = x1 * cos_vec[half:] + x2 * sin_vec[half:]

    tl.store(KeyR_ptr + b * stride_rkb + n * stride_rkn + s * stride_rks + tl.arange(0, D) * stride_rkd, y)


@triton.jit
def scatter_cache_kernel(KeyR_ptr, Value_ptr, KeyCache_ptr, ValueCache_ptr,
                         B, NKV, S, pos_ids_ptr, cache_len, stride_krb, stride_rkn, stride_rks, stride_rkd,
                         stride_vb, stride_vk, stride_vs, stride_vd,
                         stride_kcb, stride_kcn, stride_kcs, stride_kcd,
                         stride_vcb, stride_vcn, stride_vcs, stride_vcd):
    """
    For each (b, n) and s in [0..S-1], load KeyR[b, n, s, :] and Value[b, n, s, :] and write into
    KeyCache[b, n, cache_position[s], :] and ValueCache[b, n, cache_position[s], :].
    Grid: (B, NKV)
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    if (b >= B) or (n >= NKV):
        return
    # Loop over S tokens
    for s in range(0, S):
        pos = tl.load(pos_ids_ptr + b * S + s).to(tl.int32) + cache_len
        # Load rotated key and value for this (b, n, s)
        xk = tl.load(KeyR_ptr + b * stride_krb + n * stride_rkn + s * stride_rks + tl.arange(0, D) * stride_rkd)
        xv = tl.load(Value_ptr + b * stride_vb + n * stride_vk + s * stride_vs + tl.arange(0, D) * stride_vd)
        # Store into cache at index pos
        tl.store(KeyCache_ptr + b * stride_kcb + n * stride_kcn + pos * stride_kcs + tl.arange(0, D) * stride_kcd, xk)
        tl.store(ValueCache_ptr + b * stride_vcb + n * stride_vcn + pos * stride_vcs + tl.arange(0, D) * stride_vcd, xv)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # We are not using q_norm_weight/k_norm_weight (original does RMSNorm without affine). We still accept them for signature compatibility.

        B, N_q, S, D = query.shape
        B_k, N_kv, S_k, D_k = key.shape
        assert B == B_k and S == S_k and D == D_k, "Incompatible shapes"

        # 1) RMSNorm for query and key using Triton
        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch RMSNorm for query
        grid_q = (B * N_q * S,)
        rmsnorm_rows_kernel[grid_q](
            query, query_norm,
            B * N_q * S, D, rms_norm_eps,
            query.stride(0), query.stride(3),
            query_norm.stride(0), query_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # Launch RMSNorm for key
        grid_k = (B * N_kv * S,)
        rmsnorm_rows_kernel[grid_k](
            key, key_norm,
            B * N_kv * S, D, rms_norm_eps,
            key.stride(0), key.stride(3),
            key_norm.stride(0), key_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # 2) Rotate query and key using Triton (per-token cos/sin, rotation)
        # Prepare output tensors
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        # Launch query rotation: grid over (B*N_q, S)
        grid_qr = (B * N_q, S)
        rotate_and_scale_query_kernel[grid_qr](
            query_norm, query_rot,
            B, N_q, S, D, inv_freq,
            position_ids,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            num_warps=4, num_stages=2
        )

        # Launch key rotation: grid over (B*N_kv, S)
        grid_kr = (B * N_kv, S)
        rotate_and_scale_key_kernel[grid_kr](
            key_norm, key_rot,
            B, N_kv, S, D, inv_freq,
            position_ids,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            num_warps=4, num_stages=2
        )

        # 3) Scatter updates to cache using Triton
        # Note: cache_position is int64 on device, position_ids is int64; we read both in kernel. Cache tensors are (B, N_kv, max_pos, D).
        grid_scatter = (B, N_kv)
        scatter_cache_kernel[grid_scatter](
            key_rot, value, key_cache, value_cache,
            B, N_kv, S, position_ids, int(cache_position[0].item()),  # cache_len from cache_position[0]
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=4, num_stages=2
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
