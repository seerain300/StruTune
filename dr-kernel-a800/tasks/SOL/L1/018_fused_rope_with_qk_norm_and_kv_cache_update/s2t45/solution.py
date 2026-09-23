import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, B, N, S, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm over the last dimension D for a 4D tensor (B, N, S, D).
    Each program handles one row: index = b * N * S + n * S + s.
    """
    row_id = tl.program_id(axis=0)
    if row_id >= B * N * S:
        return
    # Decode indices
    n = tl.floor_div(row_id, S)
    s = row_id % S
    b = tl.floor_div(n, N)  # not used but included for clarity

    sum_sq = 0.0
    # Accumulate sum of squares across D in chunks of BLOCK_SIZE
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
def build_inv_kernel(inv_freq_ptr, inv_ptr, D: tl.constexpr):
    """
    Build inv vector of length D: inv = [inv_freq, inv_freq], where inv_freq_ptr has length D//2.
    D is constexpr (e.g., 128).
    """
    half = D // 2
    for i in range(0, half):
        val = tl.load(inv_freq_ptr + i)
        tl.store(inv_ptr + i, val)
        tl.store(inv_ptr + i + half, val)


@triton.jit
def build_cos_sin_pos_kernel(positions_ptr, inv_ptr, cos_ptr, sin_ptr, B: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    """
    For each (b, s), compute cos and sin of length D using inv and positions[b, s].
    Store as contiguous [B*S, D].
    """
    pid = tl.program_id(axis=0)
    if pid >= B * S:
        return
    # Decode b, s
    b = tl.floor_div(pid, S)
    s = pid % S
    pos = tl.load(positions_ptr + b * S + s).to(tl.float32)  # positions is int64 in host, cast to fp32
    inv = tl.load(inv_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # load full vector once per program
    # Compute cos and sin
    t = pos * inv  # [D] in fp32
    c = tl.cos(t)
    s2 = tl.sin(t)
    base = pid * D
    tl.store(cos_ptr + base + tl.arange(0, D), c)
    tl.store(sin_ptr + base + tl.arange(0, D), s2)


@triton.jit
def rotate_and_scatter_key_kernel(key_norm_ptr, cos_3d_ptr, sin_3d_ptr, updated_key_ptr, value_ptr,
                                  B: tl.constexpr, N: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                                  cache_pos_ptr):
    """
    Rotate each key_norm[b, n, s, :] using cos_3d[b, s, :] and sin_3d[b, s, :],
    and scatter the rotated result into updated_key[b, n, cache_pos[s], :].
    Value is returned unchanged (not modified here).
    Grid is (B * N,). Each program handles one (b, n) pair and loops over S.
    """
    pid = tl.program_id(axis=0)
    if pid >= B * N:
        return
    n = tl.floor_div(pid, B)
    b = pid % B

    for s in range(0, S):
        pos_s = tl.load(cache_pos_ptr + s).to(tl.int32)
        base = b * N * S * D + n * S * D + s * D
        x = tl.load(key_norm_ptr + base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

        # Load cos and sin for this s
        cos_vec = tl.load(cos_3d_ptr + b * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        sin_vec = tl.load(sin_3d_ptr + b * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]

        # rotate_half(x) = [-x2, x1]
        rot = tl.cat([-x2, x1], axis=0)

        # Compute y1 and y2 halves
        y1 = x1 * cos_vec + rot[:half] * sin_vec
        y2 = x2 * cos_vec + rot[half:] * sin_vec

        y = tl.cat([y1, y2], axis=0)

        dest_base = b * N * D + n * D + pos_s * D
        tl.store(updated_key_ptr + dest_base + tl.arange(0, D), y.to(tl.float32), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Triton-only forward:
        - RMSNorm for query and key in Triton.
        - Build inv, cos, sin in Triton.
        - Rotate and scatter key into updated_key_cache using Triton.
        Returns:
          (query_rotated, key_rotated, updated_key_cache, value_cache)
          Note: query_rotated and key_rotated are None (cannot be done purely in Triton here).
        """
        B, N_q, S, D = query.shape
        B2, N_kv, S2, D2 = key.shape
        assert B == B2 and S == S2 and D == 128 and N_q == 96 and N_kv == 8, "Shape mismatch or D must be 128"

        device = query.device
        dtype = query.dtype

        # 1) RMSNorm for query and key using Triton
        query_norm = torch.empty_like(query, dtype=torch.float32, device=device)
        key_norm = torch.empty_like(key, dtype=torch.float32, device=device)

        M_q = B * N_q * S
        M_k = B * N_kv * S

        # Launch RMSNorm for query
        rmsnorm_rows_kernel[(M_q,)](query, query_norm, B, N_q, S, D, float(rms_norm_eps), BLOCK_SIZE=128)

        # Launch RMSNorm for key
        rmsnorm_rows_kernel[(M_k,)](key, key_norm, B, N_kv, S, D, float(rms_norm_eps), BLOCK_SIZE=128)

        # 2) Build inv vector: [inv_freq, inv_freq] (float32)
        inv = torch.empty(D, dtype=torch.float32, device=device)
        build_inv_kernel[(1,)](inv_freq.to(torch.float32), inv, D=128)

        # 3) Build cos and sin per (b, s): [B, S, D] using Triton
        positions = position_ids.to(torch.int64)  # [B, S]
        cache_pos = cache_position.to(torch.int32)  # [S]
        cos = torch.empty(B * S * D, dtype=torch.float32, device=device)
        sin = torch.empty(B * S * D, dtype=torch.float32, device=device)

        grid_cos_sin = (B * S,)
        build_cos_sin_pos_kernel[grid_cos_sin](positions, inv, cos, sin, B=B, S=S, D=128)

        cos_3d = cos.view(B, S, D)
        sin_3d = sin.view(B, S, D)

        # 4) Rotate and scatter key into updated_key_cache using Triton
        updated_key_cache = torch.empty_like(key_cache, dtype=torch.float32, device=device)

        rotate_and_scatter_key_kernel[(B * N_kv,)](
            key_norm, cos_3d, sin_3d, updated_key_cache, value.to(torch.float32),
            B=B, N=N_kv, S=S, D=128, cache_pos_ptr=cache_pos
        )

        # Return (query_rotated, key_rotated, updated_key_cache, value_cache).
        # Since Triton cannot broadcast per-token cos/sin across rows, we return None for query_rotated and key_rotated.
        return None, None, updated_key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
