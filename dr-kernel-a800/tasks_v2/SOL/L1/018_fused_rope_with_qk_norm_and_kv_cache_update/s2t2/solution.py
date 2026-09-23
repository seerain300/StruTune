import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps):
    """
    RMSNorm over last dimension of shape (M, D).
    Each program handles one row (M). Computes r = sqrt(mean(x^2) + eps) and writes y = x / r.
    X_ptr, Y_ptr are 1D contiguous arrays of length M*D. eps: float32.
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return
    sum_sq = 0.0
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * D + d)
        x_f32 = x.to(tl.float32)
        sum_sq += x_f32 * x_f32
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)
    for d in range(0, D):
        x = tl.load(X_ptr + row_id * D + d)
        y = x / r
        tl.store(Y_ptr + row_id * D + d, y)


@triton.jit
def rotate_and_scale_kernel(X_ptr, Y_ptr, B, N_HEADS, S, D, inv_ptr, POS_PTR):
    """
    Per-row rotation and scaling (RoPE) applied to X_ptr -> Y_ptr.
    Grid: (B * N_HEADS * S,)
    For each row (b, head, s), compute cos/sin vectors using inv_ptr and POS_PTR[b, s],
    then apply: y = x1 * cos + rotate_half(x) * sin.
    X_ptr, Y_ptr: 1D contiguous arrays of length (B*N_HEADS*S)*D.
    inv_ptr: float32, length D, inv = [inv_freq, inv_freq].
    POS_PTR: int64, shape (B, S); we load pos from POS_PTR[b, s].
    """
    row = tl.program_id(axis=0)
    total = B * N_HEADS * S
    if row >= total:
        return
    S_i32 = tl.full((), S, tl.int32)
    N_heads_i32 = tl.full((), N_HEADS, tl.int32)
    b = row // (N_heads_i32 * S_i32)
    rem = row % (N_heads_i32 * S_i32)
    s = rem // N_heads_i32
    head = rem % N_heads_i32

    pos_val = tl.load(POS_PTR + b * S + s)  # int64
    pos_f32 = pos_val.to(tl.float32)

    # Build cos and sin vectors of length D
    cos_vec = tl.zeros((D,), dtype=tl.float32)
    sin_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        inv_d = tl.load(inv_ptr + d)  # float32
        angle = pos_f32 * inv_d
        cos_vec[d] = tl.cos(angle)
        sin_vec[d] = tl.sin(angle)

    base = row * D
    x = [tl.load(X_ptr + base + d) for d in range(0, D)]
    x_f32 = [x_i.to(tl.float32) for x_i in x]
    x1 = x_f32[:D // 2]
    x2 = x_f32[D // 2:]
    rotated_half = [-x2[0]] + [x1_i for x1_i in x1]  # rotated_half has length D//2, but we'll reconstruct y with concatenation below

    # Construct y: y1 = x1 * cos_vec[:D//2], y2 = rotated_half * sin_vec[:D//2]
    y = []
    for i in range(D // 2):
        y1_i = x1[i] * cos_vec[i]
        y2_i = (-x2[i]) * sin_vec[i]  # because rotated_half second half is -x2
        # rotate_half(x): first half is x1, second half is -x2, so y2_i corresponds to sin_vec[i] with negative x2[i]
        y.append(y1_i)
        y.append(y2_i)

    # Store results
    for d in range(0, D):
        tl.store(Y_ptr + base + d, y[d].to(tl.bfloat16))


@triton.jit
def write_key_cache_kernel(KEY_PTR, KEYCACHE_PTR, CACHE_PTR, B, N_KV, S, D):
    """
    Scatter write for key_cache:
    For each (b, nk, s), read D elements from KEY_PTR[b, nk, s, :] and write to
    key_cache[b, nk, int(CACHE_PTR[s]), :].
    Grid: (B * N_KV, S). Each program handles one (b, nk, s).
    KEY_PTR: 1D contiguous pointer of length (B*N_KV*S)*D.
    KEYCACHE_PTR: 1D contiguous pointer of length (B*N_KV*MAX_POS)*D.
    CACHE_PTR: int64 of length S, cache positions.
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (pid >= B * N_KV) or (s >= S):
        return
    b = pid // N_KV
    nk = pid % N_KV

    # Load destination row index
    row = tl.load(CACHE_PTR + s)  # int64

    # Compute linear indices
    src_lin = ((b * N_KV + nk) * S + s) * D
    dest_lin = (b * N_KV + nk) * (self.MAX_POS * D) + row * D

    offs = tl.arange(0, D)
    vals = tl.load(KEY_PTR + src_lin + offs)
    tl.store(KEYCACHE_PTR + dest_lin + offs, vals)


@triton.jit
def write_value_cache_kernel(VALUE_PTR, VALUECACHE_PTR, CACHE_PTR, B, N_KV, S, D):
    """
    Scatter write for value_cache:
    For each (b, nk, s), read D elements from VALUE_PTR[b, nk, s, :] and write to
    value_cache[b, nk, int(CACHE_PTR[s]), :].
    Same grid and logic as write_key_cache_kernel.
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (pid >= B * N_KV) or (s >= S):
        return
    b = pid // N_KV
    nk = pid % N_KV

    row = tl.load(CACHE_PTR + s)

    src_lin = ((b * N_KV + nk) * S + s) * D
    dest_lin = (b * N_KV + nk) * (self.MAX_POS * D) + row * D

    offs = tl.arange(0, D)
    vals = tl.load(VALUE_PTR + src_lin + offs)
    tl.store(VALUECACHE_PTR + dest_lin + offs, vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.half_dim = 64
        self.max_position_embeddings = 262144
        self.rms_norm_eps = 1e-6

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure CUDA device
        device = query.device
        assert device.type == 'cuda', "ModelNew requires a CUDA device for Triton kernels."

        # Contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        inv_freq = inv_freq.contiguous()

        B = query.shape[0]
        N_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        N_kv = key.shape[1]

        # 1) RMSNorm for query
        M_query = B * N_q * S
        query_norm = torch.empty_like(query, dtype=query.dtype)
        rmsnorm_rows_kernel[(M_query,)](query, query_norm, M_query, D, float(rms_norm_eps))

        # 2) RMSNorm for key
        M_key = B * N_kv * S
        key_norm = torch.empty_like(key, dtype=key.dtype)
        rmsnorm_rows_kernel[(M_key,)](key, key_norm, M_key, D, float(rms_norm_eps))

        # 3) Rotation for query
        query_rot = torch.empty_like(query_norm, dtype=query_norm.dtype)
        total_rows_query = M_query
        rotate_and_scale_kernel[(total_rows_query,)](
            query_norm, query_rot, B, N_q, S, D, inv_freq, position_ids
        )

        # 4) Rotation for key
        key_rot = torch.empty_like(key_norm, dtype=key_norm.dtype)
        total_rows_key = M_key
        rotate_and_scale_kernel[(total_rows_key,)](
            key_norm, key_rot, B, N_kv, S, D, inv_freq, position_ids
        )

        # 5) Scatter writes to caches in Triton (no torch indexing)
        # Cast cache_position to int64 for Triton loads
        cache_pos_i64 = cache_position.to(torch.int64)

        # Launch write_key_cache_kernel: grid = (B*N_kv, S)
        write_key_cache_kernel[(B * N_kv, S)](
            key_rot, key_cache, cache_pos_i64, B, N_kv, S, D, MAX_POS=self.max_position_embeddings
        )

        # Launch write_value_cache_kernel: grid = (B*N_kv, S)
        write_value_cache_kernel[(B * N_kv, S)](
            value, value_cache, cache_pos_i64, B, N_kv, S, D, MAX_POS=self.max_position_embeddings
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
