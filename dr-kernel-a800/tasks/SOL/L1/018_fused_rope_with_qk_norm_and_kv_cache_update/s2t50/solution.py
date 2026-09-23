import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm over last dimension D for M rows of a 4D tensor (B, N, S, D).
    Each program handles one row: computes r = sqrt(mean(x^2) + eps) and writes y = x / r.
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return
    sum_sq = 0.0
    # Accumulate sum of squares in fp32
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)
    # Scale and store
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        y = (x_f32 / r).to(x.dtype)
        tl.store(Y_ptr + row_id * D + offs, y, mask=mask)


@triton.jit
def build_inv_kernel(inv_freq_ptr, inv_ptr, D_HALF: tl.constexpr, D: tl.constexpr):
    """
    Triton kernel: build inv vector of length D = [inv_freq, inv_freq].
    inv_freq_ptr: [D_HALF] float32
    inv_ptr: [D] float32
    """
    idx = tl.program_id(axis=0)
    if idx >= D:
        return
    if idx < D_HALF:
        tl.store(inv_ptr + idx, tl.load(inv_freq_ptr + idx))
    else:
        tl.store(inv_ptr + idx, tl.load(inv_freq_ptr + (idx - D_HALF)))


@triton.jit
def cos_sin_rows_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr, S, D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: For each token s (grid axis 0 over S), compute cos and sin vectors of length D:
      cos[s, :] = cos(pos[s] * inv[:])
      sin[s, :] = sin(pos[s] * inv[:])
    pos_ptr: [S] int32
    inv_ptr: [D] float32
    cos_ptr: [S, D] float32 (we store per-batch by passing pointer to cos_ptr + b*S*D)
    sin_ptr: [S, D] float32
    """
    s = tl.program_id(axis=0)
    if s >= S:
        return
    pos = tl.load(pos_ptr + s).to(tl.int32)
    base = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        angle = base + pos * tl.load(inv_ptr + offs)
        c = tl.cos(angle).to(tl.float32)
        s = tl.sin(angle).to(tl.float32)
        tl.store(cos_ptr + s * D + offs, c, mask=(offs < D))
        tl.store(sin_ptr + s * D + offs, s, mask=(offs < D))


@triton.jit
def rotate_rows_1d_kernel(query_row_ptr, key_row_ptr, cos_ptr, sin_ptr, out_q_row_ptr, out_k_row_ptr, D, BLOCK_SIZE: tl.constexpr):
    """
    1D Triton kernel: rotate a single row (query_row_ptr, key_row_ptr) of length D using cos/sin vectors.
    Writes to out_q_row_ptr and out_k_row_ptr.
    Assumes cos/sin are per-token vectors of length D.
    """
    # We will launch this per (b,n,s) in host. Here, we implement a single-row rotation with provided pointers.
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        xq = tl.load(query_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xk = tl.load(key_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Split into halves
        half = D // 2
        xq1 = xq[:half]
        xq2 = xq[half:]
        xk1 = xk[:half]
        xk2 = xk[half:]
        # For key: rotate_half(key) = [xk2, -xk1]
        rot_k1 = xk2
        rot_k2 = -xk1
        # For query: rotate_half(query) = [-xq2, xq1]
        rot_q1 = xq1
        rot_q2 = -xq2
        # Compute rotated halves
        yq1 = xq1 * cos[:half] + rot_q2 * sin[:half]
        yq2 = xq2 * cos[half:] + rot_q1 * sin[half:]  # Note: we use rot_q1 which is xq1; fixed below
        # Correct: yq1 = xq1 * cos[:half] + (-xq2) * sin[:half]
        #          yq2 = xq2 * cos[half:] + xq1 * sin[half:]
        yq1 = xq1 * cos[:half] + (-xq2) * sin[:half]
        yq2 = xq2 * cos[half:] + xq1 * sin[half:]
        yk1 = xk1 * cos[:half] + rot_k2 * sin[:half]
        yk2 = xk2 * cos[half:] + rot_k1 * sin[half:]
        yq = tl.concatenate([yq1, yq2])
        yk = tl.concatenate([yk1, yk2])
        tl.store(out_q_row_ptr + offs, yq, mask=mask)
        tl.store(out_k_row_ptr + offs, yk, mask=mask)


@triton.jit
def scatter_update_kernel(key_cache_ptr, value_ptr, cache_pos_ptr, S, D):
    """
    Triton kernel: For each token s (grid axis 0 over S), read rotated key row and value row,
    and write into key_cache[b, n, cache_pos[s], :] and value_cache[b, n, cache_pos[s], :].
    Note: This kernel assumes we pass base pointers for (b, n) via pointer arithmetic in host.
    """
    s = tl.program_id(axis=0)
    if s >= S:
        return
    pos = tl.load(cache_pos_ptr + s).to(tl.int32)
    # Host will set base pointers for (b, n)
    # Placeholder: no-op to satisfy signature; actual updates are done in host loops using PyTorch.
    pass


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor,
                value_cache: torch.Tensor, cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        # Ensure head_dim=128
        D = query.shape[-1]
        assert D == 128, "This implementation assumes head_dim=128"
        D_HALF = D // 2

        B = query.shape[0]
        N_q = query.shape[1]
        S = query.shape[2]
        N_kv = key.shape[1]

        device = query.device

        # 1) RMSNorm for query and key (compute in fp32, return in original dtype)
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        M_q = B * N_q * S
        rmsnorm_rows_kernel[(M_q,)](query, query_norm, M_q, D, rms_norm_eps, BLOCK_SIZE=64, num_warps=4)
        M_k = B * N_kv * S
        rmsnorm_rows_kernel[(M_k,)](key, key_norm, M_k, D, rms_norm_eps, BLOCK_SIZE=64, num_warps=4)

        # 2) Build inv vector of length D = [inv_freq, inv_freq]
        inv = torch.empty(D, dtype=torch.float32, device=device)
        build_inv_kernel[(D,)](inv_freq.to(torch.float32), inv, D_HALF, D)

        # 3) Compute cos and sin per token s: [B, S, D] in fp32
        pos = position_ids.to(torch.int32)
        cos_buffers = [torch.empty(S * D, dtype=torch.float32, device=device) for _ in range(B)]
        sin_buffers = [torch.empty(S * D, dtype=torch.float32, device=device) for _ in range(B)]
        cos_ptrs = [cos_buffers[b] for b in range(B)]
        sin_ptrs = [sin_buffers[b] for b in range(B)]

        cos_sin_rows_kernel[(S,)](pos.view(-1), inv, cos_ptrs[0], sin_ptrs[0], S, D, BLOCK_SIZE=64, num_warps=4)
        # Construct cos/sin per batch
        cos_list = []
        sin_list = []
        for b in range(B):
            cos_list.append(cos_ptrs[b].view(S, D))
            sin_list.append(sin_ptrs[b].view(S, D))

        # 4) Rotate rows: rotated_query and rotated_key
        rotated_query = torch.empty_like(query_norm)
        rotated_key = torch.empty_like(key_norm)
        for b in range(B):
            for n in range(N_q):
                # For each token s, read query_norm[b, n, s], rotate, and store
                for s in range(S):
                    q_row = query_norm[b, n, s]
                    out_q_row = torch.empty_like(q_row)
                    # cos/sin for this token s
                    cos_s = cos_list[b][s]
                    sin_s = sin_list[b][s]
                    rotate_rows_1d_kernel[(1,)](q_row, q_row, cos_s, sin_s, out_q_row, out_q_row, D, BLOCK_SIZE=64, num_warps=4)
                    rotated_query[b, n, s] = out_q_row

            for n in range(N_kv):
                for s in range(S):
                    k


def run(*args):
    return ModelNew()(*args)
