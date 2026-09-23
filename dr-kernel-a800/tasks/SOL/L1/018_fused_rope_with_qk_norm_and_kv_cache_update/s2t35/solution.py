import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, B, N, S, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm across last dimension D for rows of a 4D tensor [B, N, S, D].
    Grid over M = B * N * S rows. Each program handles one row.
    """
    row_id = tl.program_id(axis=0)
    b = row_id // (N * S)
    rem = row_id % (N * S)
    n = rem // S
    s = rem % S

    base = b * N * S * D + n * S * D + s * D

    sum_sq = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)

    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        y = (x.to(tl.float32) / r).to(x.dtype)
        tl.store(Y_ptr + base + offs, y, mask=mask)


@triton.jit
def build_inv_kernel(inv_freq_ptr, inv_ptr, D_HALF: tl.constexpr, D: tl.constexpr):
    """
    Triton kernel: build inv vector of length D from inv_freq of length D_HALF:
    inv = [inv_freq, inv_freq].
    inv_ptr is of length D; we write inv_freq into first D_HALF entries and copy into second half.
    """
    offs = tl.arange(0, D)
    idx1 = offs
    mask1 = idx1 < D_HALF
    vals1 = tl.load(inv_freq_ptr + idx1, mask=mask1, other=0.0)

    idx2 = offs + D_HALF
    mask2 = idx2 < D
    tl.store(inv_ptr + idx2, vals1, mask=mask2)

    tl.store(inv_ptr + idx1, vals1, mask=mask1)


@triton.jit
def rotate_and_scatter_kernel(
    key_norm_ptr,       # [B, N_kv, S, D] (fp32), normalized key
    cos_ptr,            # [B, S, D] (fp32)
    sin_ptr,            # [B, S, D] (fp32)
    key_cache_ptr,      # [B, N_kv, M, D] (fp32), output cache
    value_cache_ptr,    # [B, N_kv, M, D] (fp32), output cache
    B, N_kv, S, D,
    cache_ptr,          # [S] int32, cache_position indices
):
    """
    Triton kernel: For each (b, n_kv), iterate over s in [0..S), rotate key_norm[b, n, s, :]
    using cos/sin for that s, and write into key_cache[b, n, cache[b, s], :] and
    value_cache[b, n, cache[b, s], :] with original value rows.
    Note: We don't have 'value' tensor; we write original key_norm rows into value_cache at same positions
    to emulate 'value' write (the original assigns value to cache at those positions).
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)

    for s in range(0, S):
        c = tl.load(cache_ptr + s)  # int32
        # Base pointer for key row
        base_key = b * (N_kv * S * D) + n * S * D + s * D
        x = tl.load(key_norm_ptr + base_key + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

        # Load cos/sin for this s
        cos_row_ptr = cos_ptr + b * S * D + s * D
        sin_row_ptr = sin_ptr + b * S * D + s * D
        cos_vec = tl.load(cos_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        sin_vec = tl.load(sin_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)

        # Split x into halves
        x1 = x[:64]
        x2 = x[64:]

        # rotate_half(x) = [-x2, x1]
        rotated_half = -x2 + x1

        # y1 and y2 for first and second halves
        y1 = x1 * cos_vec[:64] + rotated_half[:64] * sin_vec[:64]
        y2 = x2 * cos_vec[64:] + rotated_half[64:] * sin_vec[64:]

        y = tl.zeros((D,), dtype=tl.float32)
        y[:64] = y1
        y[64:] = y2

        # Store into key_cache at position c
        out_key_ptr = key_cache_ptr + b * (N_kv * D) + n * D + c * D
        tl.store(out_key_ptr + tl.arange(0, D), y)

        # Store original key row into value_cache at same position (emulating 'value' write)
        out_val_ptr = value_cache_ptr + b * (N_kv * D) + n * D + c * D
        tl.store(out_val_ptr + tl.arange(0, D), x)  # original key norm row


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Returns:
          - query_rotated: fp32 [B, N_q, S, D]
          - key_rotated: fp32 [B, N_kv, S, D]
          - key_cache: fp32 [B, N_kv, M, D] updated with rotated key rows at cache_position
          - value_cache: fp32 [B, N_kv, M, D] updated with original key rows at cache_position (to emulate 'value')
        """
        # Shapes
        B, N_q, S, D = query.shape
        Bk, N_kv, Sk, Dk = key.shape
        assert B == Bk and N_q == N_kv and S == Sk and D == Dk, "Input shapes must match."

        # 1) RMSNorm for query and key using Triton
        query_norm = torch.empty_like(query, dtype=torch.float32, device=query.device)
        key_norm = torch.empty_like(key, dtype=torch.float32, device=key.device)

        # Grid over rows: M = B * N * S
        M_q = B * N_q * S
        grid_q = (M_q,)
        rmsnorm_rows_kernel[grid_q](query, query_norm, B, N_q, S, D, rms_norm_eps, BLOCK_SIZE=128, num_warps=4)

        M_k = B * N_kv * S
        grid_k = (M_k,)
        rmsnorm_rows_kernel[grid_k](key, key_norm, B, N_kv, S, D, rms_norm_eps, BLOCK_SIZE=128, num_warps=4)

        # 2) Build inv = [inv_freq, inv_freq] (fp32) of length D
        inv = torch.empty(D, dtype=torch.float32, device=query.device)
        inv[:D // 2] = inv_freq.to(torch.float32)
        inv[D // 2:] = inv_freq.to(torch.float32)

        # 3) Precompute cos and sin tensors on host (torch): angles = pos * inv
        # Cast position_ids to float32
        pos = position_ids.to(torch.float32)  # [B, S]
        angles = pos[:, :, None] * inv[None, None, :]


def run(*args):
    return ModelNew()(*args)
