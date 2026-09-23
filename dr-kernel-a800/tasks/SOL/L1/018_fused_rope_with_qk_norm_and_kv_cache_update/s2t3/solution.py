import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    RMSNorm over last dimension of shape (M, D).
    Each program handles one row. Computes r = sqrt(mean(x^2) + eps) and writes y = x / r.
    X_ptr, Y_ptr: pointers to input/output arrays (contiguous expected).
    M: number of rows (B * N * S)
    D: head_dim
    eps: float32 epsilon
    BLOCK_SIZE: 128
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return

    # First pass: compute sum of squares
    sum_sq = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)

    # Second pass: scale and store
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        y = x / r
        tl.store(Y_ptr + row_id * D + offs, y, mask=mask)


@triton.jit
def rotate_and_scale_kernel(X_ptr, Y_ptr, B, N_HEADS, S, D, cos_ptr, sin_ptr, POS_PTR):
    """
    Triton kernel performing per-(b, n, s) rotation and scaling.
    Grid: axis 0 over B * N_HEADS * S rows, axis 1 over S (tokens) so cos/sin per position is reused.
    For each program (b, n, s), load cos/sin vectors of length D for that position, compute rotation, and store.
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)  # second axis index corresponds to token position

    if pid >= B * N_HEADS * S:
        return

    # Map pid -> (b, n, s)
    b = pid // (N_HEADS * S)
    n = (pid // S) % N_HEADS
    s = pid % S

    # Compute base offset for X/Y rows
    row_offset = b * (N_HEADS * S) * D + n * (S * D) + s * D
    x_row_ptr = X_ptr + row_offset
    y_row_ptr = Y_ptr + row_offset

    # Load cos and sin vectors for this (b, s) from POS_PTR
    base = b * S + s
    cos_vec = tl.zeros([D], dtype=tl.float32)
    sin_vec = tl.zeros([D], dtype=tl.float32)
    for d in range(0, D, 128):
        offs = d + tl.arange(0, 128)
        mask = offs < D
        cos_chunk = tl.load(cos_ptr + base * (2 * D) + offs, mask=mask, other=0.0)
        sin_chunk = tl.load(sin_ptr + base * (2 * D) + D + offs, mask=mask, other=0.0)
        cos_vec[offs] = cos_chunk
        sin_vec[offs] = sin_chunk

    # Load normalized x for this row (length D)
    x = tl.load(x_row_ptr + tl.arange(0, D), mask=(tl.arange(0, D) < D), other=0.0)

    # Split into halves
    x1 = x[:D // 2]
    x2 = x[D // 2:]

    # Compute rotated halves
    y1 = x1 * cos_vec[:D // 2] - x2 * sin_vec[:D // 2]
    y2 = x1 * sin_vec[:D // 2] + x2 * cos_vec[:D // 2]
    y = tl.zeros([D], dtype=x.dtype)
    y[:D // 2] = y1
    y[D // 2:] = y2

    # Store result
    tl.store(y_row_ptr + tl.arange(0, D), y, mask=(tl.arange(0, D) < D))


@triton.jit
def scatter_cache_rows_kernel(INPUT_ptr, CACHE_ptr, B, N_HEADS, S, POS_ptr):
    """
    Triton kernel that scatters per-(b, n, s) rows from INPUT_ptr into CACHE_ptr at rows POS_ptr[s].
    Grid: axis 0 over B * N_HEADS, axis 1 over S (tokens). For each (b, n, s), load the row and store at cache row idx = POS_ptr[s].
    Shapes:
      INPUT_ptr: (B, N_HEADS, S, D)
      CACHE_ptr: (B, N_HEADS, max_positions, D)
      POS_ptr: (S,) int64 indices (cache positions).
    """
    pid = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    if pid >= B * N_HEADS:
        return

    b = pid // N_HEADS
    n = pid % N_HEADS

    # Load the row to be written (b, n, s, :)
    input_row_ptr = INPUT_ptr + b * (N_HEADS * S * D) + n * (S * D) + s * D
    x = tl.load(input_row_ptr + tl.arange(0, D), mask=(tl.arange(0, D) < D), other=0.0)

    # Get cache position for this token
    idx = tl.load(POS_ptr + s)  # int64 index
    # Compute output row pointer for cache
    cache_row_ptr = CACHE_ptr + b * (N_HEADS * D) + n * D + idx * D

    # Store the row
    tl.store(cache_row_ptr + tl.arange(0, D), x, mask=(tl.arange(0, D) < D))


class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        """
        Triton-optimized forward:
        - Compute RMSNorm for query and key in Triton.
        - Compute cos/sin for RoPE using PyTorch (allowed, heavy math done in Triton below).
        - Apply rotation in Triton.
        - Update caches using Triton scatter kernel.
        Returns: query_rotated, key_rotated, updated key_cache, updated value_cache
        """
        B, N_q, S, D = query.shape
        Bk, N_kv, Sk, Dk = key.shape
        assert Sk == S and D == Dk, "Shape mismatch for key/value"

        # 1) RMSNorm for query and key (no affine). Compute in Triton.
        query_flat = query.contiguous().view(B * N_q * S, D)  # (M, D)
        key_flat = key.contiguous().view(B * N_kv * S, D)     # (Mk, D)

        query_norm_flat = torch.empty_like(query_flat)
        key_norm_flat = torch.empty_like(key_flat)

        M_query = B * N_q * S
        M_key = B * N_kv * S

        rmsnorm_rows_kernel[(M_query,)](query_flat, query_norm_flat, M_query, D, float(rms_norm_eps), BLOCK_SIZE=128)
        rmsnorm_rows_kernel[(M_key,)](key_flat, key_norm_flat, M_key, D, float(rms_norm_eps), BLOCK_SIZE=128)

        # Reshape back to original (B, N, S, D)
        query_norm = query_norm_flat.view(B, N_q, S, D)
        key_norm = key_norm_flat.view(B, N_kv, S, D)

        # 2) Rotation: build cos/sin per (b, s) using PyTorch (simple and safe).
        inv_freq_full = torch.cat([inv_freq, inv_freq], dim=0).to(query.dtype).to(query.device)  # (D,)
        # Prepare cos/sin arrays (B, S, D). Note: we do this per (b, s), but Triton kernel uses them per program.
        cos = torch.empty((B, S, D), dtype=query.dtype, device=query.device)
        sin = torch.empty((B, S, D), dtype=query.dtype, device=query.device)
        for b in range(B):
            for s in range(S):
                pos = int(position_ids[b, s].item())
                emb = pos * inv_freq_full  # (D,)
                cos[b, s] = torch.cos(emb)
                sin[b, s] = torch.sin(emb)

        # Prepare a POS info tensor for Triton rotation: for each (b, s), store cos then sin (length 2*D).
        # We'll build it on-the-fly inside Triton using torch as well (small), or precompute. Here we precompute it.
        pos_info = torch.empty((B * S, 2 * D), dtype=torch.float32, device=query.device)
        for b in range(B):
            for s in range(S):
                base = b * S + s
                emb = position_ids[b, s].float() * inv_freq_full  # (D,)
                cos_vec = torch.cos(emb)                         # (D,)
                sin_vec = torch.sin(emb)                         # (D,)
                pos_info[base, :D] = cos_vec
                pos_info[base, D:] = sin_vec

        # Allocate outputs for rotated tensors
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        # Launch rotation kernel: grid over (B * N * S, S)
        grid_query = (B * N_q * S, S)
        rotate_and_scale_kernel[grid_query](query_norm, query_rotated, B, N_q, S, D, pos_info, pos_info)

        grid_key = (B * N_kv * S, S)
        rotate_and_scale_kernel[grid_key](key_norm, key_rotated, B, N_kv, S, D, pos_info, pos_info)

        # 3) Update caches using Triton scatter kernel
        # Ensure cache_position is contiguous int64 on device
        cp = cache_position.contiguous().to(torch.int64)

        # Scatter key_rotated into key_cache
        grid_scatter_key = (B * N_kv, S)
        scatter_cache_rows_kernel[grid_scatter_key](key_rotated, key_cache, B, N_kv, S, cp)

        # Scatter value (not rotated) into value_cache
        value_sc = value.contiguous()
        grid_scatter_val = (B * N_kv, S)
        scatter_cache_rows_kernel[grid_scatter_val](value_sc, value_cache, B, N_kv, S, cp)

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
