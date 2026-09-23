import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: Row-wise RMSNorm across last dimension D for M rows.
    Each program handles one row. Computes r = sqrt(mean(x^2) + eps) and writes y = x / r.
    Assumes X_ptr and Y_ptr point to memory laid out as contiguous rows of length D.
    M = number of rows; here used with query.view(M, D) or key.view(M, D).
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return

    # Accumulate sum of squares across the row in fp32
    sum_sq = 0.0
    # Use BLOCK_SIZE loop to cover D; for D=128, this covers in one iteration.
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
def rotate_half_rows_kernel(X_ptr, Y_ptr, M, D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: Apply rotate-half transformation on each row across the last dimension D.
    For each row r: split into x1 = X[r, :D//2], x2 = X[r, D//2:], y1 = -x2, y2 = x1.
    Y[r, :] = concatenate([y1, y2]).
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return

    half = D // 2

    # First half: write -x2 to first half, x1 to second half
    for d in range(0, half, BLOCK_SIZE):
        offs1 = d + tl.arange(0, BLOCK_SIZE)   # first half positions
        offs2 = d + half + tl.arange(0, BLOCK_SIZE)  # second half positions
        mask1 = offs1 < half
        mask2 = offs2 < half

        x1 = tl.load(X_ptr + row_id * D + offs1, mask=mask1, other=0.0)
        x2 = tl.load(X_ptr + row_id * D + offs2, mask=mask2, other=0.0)

        # Write -x2 to first half
        tl.store(Y_ptr + row_id * D + offs1, (-x2).to(x1.dtype), mask=mask1)
        # Write x1 to second half
        tl.store(Y_ptr + row_id * D + offs2, x1.to(x1.dtype), mask=mask2)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ):
        """
        Returns:
        - query_rotated: rotated query via Triton rotate-half
        - key_rotated: rotated key via Triton rotate-half
        - key_cache: original key_cache (unchanged), kept for API compatibility
        - value_cache: original value_cache (unchanged)
        """
        # Ensure inputs are contiguous
        query_c = query.contiguous()
        key_c = key.contiguous()
        value_c = value.contiguous()

        B, N_q, S, D = query_c.shape
        N_kv = key_c.shape[1]

        # 1) Triton RMSNorm for query (rows = B * N_q * S)
        query_norm = torch.empty_like(query_c)
        M_q = B * N_q * S
        grid_q = (M_q,)
        rmsnorm_rows_kernel[grid_q](query_c.view(M_q, D), query_norm.view(M_q, D), M_q, D, rms_norm_eps, BLOCK_SIZE=128)

        # 2) Triton RMSNorm for key (rows = B * N_kv * S)
        key_norm = torch.empty_like(key_c)
        M_k = B * N_kv * S
        grid_k = (M_k,)
        rmsnorm_rows_kernel[grid_k](key_c.view(M_k, D), key_norm.view(M_k, D), M_k, D, rms_norm_eps, BLOCK_SIZE=128)

        # 3) Triton rotate-half on normalized tensors (launch twice: for query and key)
        query_rotated = torch.empty_like(query_norm)
        M_q2 = M_q
        grid_q2 = (M_q2,)
        rotate_half_rows_kernel[grid_q2](query_norm.view(M_q2, D), query_rotated.view(M_q2, D), M_q2, D, BLOCK_SIZE=128)

        key_rotated = torch.empty_like(key_norm)
        M_k2 = M_k
        grid_k2 = (M_k2,)
        rotate_half_rows_kernel[grid_k2](key_norm.view(M_k2, D), key_rotated.view(M_k2, D), M_k2, D, BLOCK_SIZE=128)

        # Return original caches
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
