import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm over last dimension D for M rows. X_ptr, Y_ptr point to tensors
    of shape (M, D), where M = B * N * S. Each program handles one row.
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return
    sum_sq = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)
    # Write scaled output
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        y = (x.to(tl.float32) / r).to(x.dtype)
        tl.store(Y_ptr + row_id * D + offs, y, mask=mask)


@triton.jit
def build_inv_kernel(inv_freq_ptr, inv_ptr, D_HALF: tl.constexpr, D: tl.constexpr):
    """
    Build inv vector of length D: inv = [inv_freq, inv_freq] where inv_freq has length D_HALF.
    inv_ptr[0:D_HALF] = inv_freq_ptr[0:D_HALF]; inv_ptr[D_HALF:D] = inv_freq_ptr[0:D_HALF].
    """
    idx = tl.program_id(axis=0)
    if idx >= D:
        return
    if idx < D_HALF:
        val = tl.load(inv_freq_ptr + idx)
        tl.store(inv_ptr + idx, val)
    else:
        src = idx - D_HALF
        val = tl.load(inv_freq_ptr + src)
        tl.store(inv_ptr + idx, val)


@triton.jit
def cos_sin_rows_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr, S, D, BLOCK_SIZE: tl.constexpr):
    """
    For each token position s in [0, S), compute cos and sin vectors of length D from pos[s] and inv.
    pos_ptr: int32 [S]
    inv_ptr: float32 [D]
    cos_ptr, sin_ptr: float32 [S * D] (we write into them; no input)
    """
    s = tl.program_id(axis=0)  # we launch grid=(S,)
    if s >= S:
        return
    pos = tl.load(pos_ptr + s)  # int32
    # Compute cos and sin vectors of length D
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        angle = inv_ptr[offs] * pos
        c = tl.cos(angle)
        s = tl.sin(angle)
        # store into flattened cos_ptr[s*D:(s+1)*D]
        base = s * D
        tl.store(cos_ptr + base + offs, c, mask=mask)
        tl.store(sin_ptr + base + offs, s, mask=mask)


@triton.jit
def rotate_rows_1d_kernel(x_ptr, y_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr, D, BLOCK_SIZE: tl.constexpr):
    """
    Rotate a single 1D row of length D using 2 halfs:
    x1 = x[0:D_HALF], x2 = x[D_HALF:D]
    cos1, sin1 for first half, cos2, sin2 for second half.
    y1 = x1 * cos1 + (-x2) * sin1
    y2 = x2 * cos2 + x1 * sin2
    y = concat([y1, y2]).
    x_ptr, y_ptr point to row base (length D).
    cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr are of length D_HALF.
    """
    # We assume y_ptr is preallocated of length D; we write into it.
    D_HALF = D // 2
    for d in range(0, D_HALF, BLOCK_SIZE):
        offs1 = d + tl.arange(0, BLOCK_SIZE)
        mask1 = offs1 < D_HALF
        x1 = tl.load(x_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + (offs1 + D_HALF), mask=mask1, other=0.0).to(tl.float32)
        c1 = tl.load(cos1_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
        s1 = tl.load(sin1_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
        c2 = tl.load(cos2_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
        s2 = tl.load(sin2_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
        y1 = x1 * c1 + (-x2) * s1
        y2 = x2 * c2 + x1 * s2
        # Store y1 and y2 into y_ptr
        tl.store(y_ptr + offs1, y1.to(tl.float32).to(tl.bfloat16), mask=mask1)
        tl.store(y_ptr + (offs1 + D_HALF), y2.to(tl.float32).to(tl.bfloat16), mask=mask1)


@triton.jit
def scatter_update_kernel(
    rotated_key_ptr,  # [B, N_kv, S, D]
    value_ptr,        # [B, N_kv, S, D]
    key_cache_ptr,    # [B, N_kv, max_pos, D]
    value_cache_ptr,  # [B, N_kv, max_pos, D]
    cache_pos_ptr,    # [S], int32
    B: tl.constexpr,
    N_kv: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
):
    """
    For each (b, n, s), read rotated_key[b, n, s, :] and value[b, n, s, :], and write them
    into key_cache[b, n, cache_pos[s], :] and value_cache[b, n, cache_pos[s], :].
    Grid over (B, N_kv, S).
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    if (b >= B) or (n >= N_kv) or (s >= S):
        return
    pos = tl.load(cache_pos_ptr + s)  # int32 index in [0, max_pos)
    # Compute base offsets
    src_row_offset = (b * N_kv + n) * S + s  # for src pointers, row index within [B, N_kv, S]
    # Load rotated key and value rows
    # rotated_key_ptr is [B, N_kv, S, D], row base = b * (N_kv * S * D) + src_row_offset * D
    # We can simply index using (b, n, s) triplet with strides. For simplicity, use that layout.
    # We assume rotated_key_ptr is laid out as (B, N_kv, S, D) contiguous.
    # Similarly for value_ptr.
    # We need actual pointer arithmetic for rows:
    # For a tensor of shape (B, N, S, D), row index for (b, n, s) is ((b * N + n) * S + s) * D
    row_idx = ((b * N_kv + n) * S + s) * D
    key_row_ptr = rotated_key_ptr + row_idx
    val_row_ptr = value_ptr + row_idx

    # Prepare destination pointers
    dst_b = b
    dst_n = n
    dst_pos = pos
    dst_row_base = ((dst_b * N_kv + dst_n) * max_pos + dst_pos) * D
    # Note: We don't need max_pos in kernel; pos comes from cache_pos_ptr. But Triton needs integer.
    # We can't use max_pos here; we assume pos is valid (0 <= pos < max_pos) as per original logic.
    dst_key_ptr = key_cache_ptr + dst_row_base
    dst_val_ptr = value_cache_ptr + dst_row_base

    # Copy D elements
    for d in range(0, D, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < D
        k = tl.load(key_row_ptr + offs, mask=mask, other=0.0)
        v = tl.load(val_row_ptr + offs, mask=mask, other=0.0)
        tl.store(dst_key_ptr + offs, k, mask=mask)
        tl.store(dst_val_ptr + offs, v, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, inv_freq, rms_norm_eps):
        device = query.device
        B, N_q, S, D = query.shape
        Bk, N_kv, Sk, Dk = key.shape
        assert Bk == B and N_kv == 128 // 2 and Sk == S and Dk == D
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        inv_freq = inv_freq.contiguous()

        # 1) RMSNorm for query and key
        M_q = B * N_q * S
        query_norm = torch.empty_like(query)
        rmsnorm_rows_kernel[(M_q,)](query, query_norm, M_q, D, rms_norm_eps, BLOCK_SIZE=64, num_warps=4)
        M_k = B * N_kv * S
        key_norm = torch.empty_like(key)
        rmsnorm_rows_kernel[(M_k,)](key, key_norm, M_k, D, rms_norm_eps, BLOCK_SIZE=64, num_warps=4)

        # 2) Build inv vector: [D_HALF, D]
        D_HALF = D // 2
        inv = torch.empty(D, dtype=torch.float32, device=device)
        build_inv_kernel[(D,)](inv_freq.to(torch.float32), inv, D_HALF, D, num_warps=4)

        # 3) Compute cos and sin per token s: [B, S, D] in fp32
        # Cast position_ids to int32 for Triton
        pos = position_ids.to(torch.int32)
        # We need per-batch buffers for cos and sin
        cos_buffers = [torch.empty(S * D, dtype=torch.float32, device=device) for _ in range(B)]
        sin_buffers = [torch.empty(S * D, dtype=torch.float32, device=device) for _ in range(B)]
        cos_ptrs = [cos_buffers[b] for b in range(B)]
        sin_ptrs = [sin_buffers[b] for b in range(B)]
        # Launch kernel for each s
        for s in range(S):
            # pos is 1D [B, S], we pass pos.view(-1) and rely on indexing per batch by using a grid over (S,)
            # Triton supports per-instance program_id; we can pass pos as 1D [B*S] or [S] and index by program_id.
            cos_sin_rows_kernel[(S,)](pos.view(-1), inv, cos_ptrs[0], sin_ptrs[0], S, D, BLOCK_SIZE=64, num_warps=4)
        # Now cos_buffers[b] and sin_buffers[b] contain cos and sin for each s in that batch.
        # Create cos_list and sin_list per batch
        cos_list = []
        sin_list = []
        for b in range(B):
            cos_list.append(cos_buffers[b].view(S, D))
            sin_list.append(sin_buffers[b].view(S, D))

        # 4) Rotate rows: rotated_query and rotated_key
        rotated_query = torch.empty_like(query_norm, dtype=torch.bfloat16, device=device)
        rotated_key = torch.empty_like(key_norm, dtype=torch.bfloat16, device=device)
        # For each row: launch rotate_rows_1d_kernel
        for b in range(B):
            for n in range(N_q):
                for s in range(S):
                    q_row = query_norm[b, n, s]  # [D] tensor
                    out_q_row = torch.empty(D, dtype=torch.bfloat16, device=device)
                    cos_s = cos_list[b][s].to(torch.float32)
                    sin_s = sin_list[b][s].to(torch.float32)
                    # Split cos/sin into halves
                    cos1 = cos_s[:D_HALF]
                    sin1 = sin_s[:D_HALF]
                    cos2 = cos_s[D_HALF:]
                    sin2 = sin_s[D_HALF:]
                    rotate_rows_1d_kernel[(1,)](q_row.to(torch.float32), out_q_row.to(torch.float32),
                                                cos1, sin1, cos2, sin2, D, BLOCK_SIZE=64, num_warps=4)
                    rotated_query[b, n, s] = out_q_row

            for n in range(N_kv):
                for s in range(S):
                    k_row = key_norm[b, n, s]  # [D] tensor
                    out_k_row = torch.empty(D, dtype=torch.bfloat16, device=device)
                    cos_s = cos_list[b][s].to(torch.float32)
                    sin_s = sin_list[b][s].to(torch.float32)
                    cos1 = cos_s[:D_HALF]
                    sin1 = sin_s[:D_HALF]
                    cos2 = cos_s[D_HALF:]
                    sin2 = sin_s[D_HALF:]
                    rotate_rows_1d_kernel[(1,)](k_row.to(torch.float32), out_k_row.to(torch.float32),
                                                cos1, sin1, cos2, sin2, D, BLOCK_SIZE=64, num_warps=4)
                    rotated_key[b, n, s] = out_k_row

        # 5) Scatter updates: key_cache and value_cache
        # Convert cache_position to int32
        cache_pos_i32 = cache_position.to(torch.int32)
        max_pos = key_cache.shape[2]  # not needed explicitly in kernel; pos comes from cache_pos_i32
        # Launch scatter kernel
        scatter_update_kernel[(B, N_kv, S)](
            rotated_key, value, key_cache, value_cache, cache_pos_i32,
            B=B, N_kv=N_kv, S=S, D=D, num_warps=4
        )

        # Return results
        return rotated_query, rotated_key, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
