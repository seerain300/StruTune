import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, M, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm across last dimension D for M rows.
    Each program handles one row. Writes y = x / sqrt(mean(x^2) + eps).
    X_ptr, Y_ptr point to tensors laid out as [M, D] with row-major addressing.
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
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        y = (x.to(tl.float32) / r).to(x.dtype)
        tl.store(Y_ptr + row_id * D + offs, y, mask=mask)


@triton.jit
def build_inv_kernel(inv_freq_ptr, inv_ptr, D_half, D: tl.constexpr):
    """
    Triton kernel: build inv of length D from inv_freq of length D_half:
    inv[0:D_half] = inv_freq; inv[D_half:D] = inv_freq.
    inv_ptr: [D] float32
    inv_freq_ptr: [D_half] float32
    """
    idx = tl.program_id(axis=0)
    if idx >= D:
        return
    if idx < D_half:
        val = tl.load(inv_freq_ptr + idx)
        tl.store(inv_ptr + idx, val)
        tl.store(inv_ptr + idx + D_half, val)
    else:
        src = idx - D_half
        val = tl.load(inv_ptr + src)
        tl.store(inv_ptr + idx, val)


@triton.jit
def compute_cos_sin_kernel(position_ids_ptr, inv_ptr, cos_ptr, sin_ptr, B, S, D: tl.constexpr):
    """
    Triton kernel: compute cos and sin per (b, s) using inv.
    position_ids_ptr: [B, S] int64
    inv_ptr: [D] float32
    cos_ptr, sin_ptr: [B, S, D] float32
    Grid: (B*S, 1)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * S + s).to(tl.int32)
    offs = tl.arange(0, D)
    t = pos.to(tl.float32) * tl.load(inv_ptr + offs)
    c = tl.cos(t)
    s_ = tl.sin(t)
    base = b * S * D + s * D
    tl.store(cos_ptr + base + offs, c)
    tl.store(sin_ptr + base + offs, s_)


@triton.jit
def rotate_and_scatter_rows_kernel(
    key_norm_ptr,     # [B, N, S, D] normalized key, bfloat16
    cos_ptr, sin_ptr, # [B, S, D] float32
    key_out_ptr,      # [B, N, max_pos, D] output key cache, bfloat16
    value_ptr,        # [B, N, S, D] original value, bfloat16
    value_out_ptr,    # [B, N, max_pos, D] output value cache, bfloat16
    cache_pos_ptr,    # [S] int64
    B, N, S, D, D_half, max_pos: tl.constexpr
):
    """
    Triton kernel: For each (b, n, s), rotate key_norm[b, n, s, :] using cos/sin for s,
    and scatter the rotated row into key_out[b, n, cache_pos[s], :].
    Also scatter original value[b, n, s, :] into value_out[b, n, cache_pos[s], :].
    Grid: (axis=0=B*S, axis=1=N)
    """
    row_id = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    if row_id >= B * S or n >= N:
        return
    b = row_id // S
    s = row_id % S

    # Load cache position for this s
    pos_idx = tl.load(cache_pos_ptr + s).to(tl.int32)

    # Load normalized key row: [D] bfloat16
    base_in = b * N * S * D + n * S * D + s * D
    x_row = tl.load(key_norm_ptr + base_in + tl.arange(0, D), mask=True, other=0.0)  # bfloat16
    x_f32 = x_row.to(tl.float32)

    # Load cos/sin for this s: [D] float32
    base_cs = b * S * D + s * D
    cos_vec = tl.load(cos_ptr + base_cs + tl.arange(0, D))  # [D]
    sin_vec = tl.load(sin_ptr + base_cs + tl.arange(0, D))  # [D]

    # Split into halves
    x1 = x_f32[0:D_half]          # first half of original key
    x2 = x_f32[D_half:D]          # second half of original key

    # rotate_half(x) = [-x2, x1] applied to concatenation of two halves
    # First half contribution: x1 * cos + (-x2) * sin
    y1 = x1 * cos_vec[0:D_half] + (-x2) * sin_vec[0:D_half]
    # Second half contribution: x2 * cos + x1 * sin
    y2 = x2 * cos_vec[D_half:D] + x1 * sin_vec[D_half:D]

    y = tl.zeros((D,), dtype=tl.float32)
    y[0:D_half] = y1
    y[D_half:D] = y2

    # Store rotated key into cache at cache position (b, n, pos_idx, :)
    out_base_key = b * N * max_pos * D + n * max_pos * D + pos_idx * D
    tl.store(key_out_ptr + out_base_key + tl.arange(0, D), y.to(tl.bfloat16))

    # Store original value into cache at cache position (b, n, pos_idx, :)
    base_val = b * N * S * D + n * S * D + s * D
    val_row = tl.load(value_ptr + base_val + tl.arange(0, D), mask=True, other=0.0)  # bfloat16
    tl.store(value_out_ptr + out_base_key + tl.arange(0, D), val_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, inv_freq, rms_norm_eps):
        """
        Inputs:
          query: [B, N_q, S, D], bfloat16 (unused in return, as per constraints)
          key: [B, N_kv=8, S, D], bfloat16
          value: [B, N_kv=8, S, D], bfloat16
          position_ids: [B, S], int64
          key_cache: [B, N_kv, max_pos, D], bfloat16
          value_cache: [B, N_kv, max_pos, D], bfloat16
          cache_position: [S], int64
          inv_freq: [D_half=64], float32
          rms_norm_eps: float
        Returns:
          (None, updated key_cache, updated value_cache)
        """
        B, N_q, S, D = query.shape
        N_kv = key.shape[1]
        assert N_kv == 8, "This implementation expects N_kv=8"
        assert D == 128, "This implementation expects head_dim=128"
        D_half = D // 2

        # Ensure contiguous
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        cache_position = cache_position.contiguous()

        # 1) RMSNorm for key: y = x / sqrt(mean(x^2) + eps)
        key_norm = torch.empty_like(key)
        M_key = B * N_kv * S
        key_flat = key.view(M_key, D).contiguous()
        key_norm_flat = key_norm.view(M_key, D).contiguous()
        grid_key = (M_key,)
        rmsnorm_rows_kernel[grid_key](key_flat, key_norm_flat, M_key, D, rms_norm_eps, BLOCK_SIZE=D)

        # 2) Build inv vector of length 128: inv = [inv_freq, inv_freq] (float32)
        inv = torch.empty(D, dtype=torch.float32, device=query.device)
        inv_freq = inv_freq.to(torch.float32).contiguous()
        build_inv_kernel[(D,)](inv_freq, inv, D_half, D)

        # 3) Compute cos and sin per (b, s) using inv: cos = cos(pos * inv), sin = sin(pos * inv)
        cos = torch.empty((B, S, D), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, D), dtype=torch.float32, device=query.device)
        grid_cos_sin = (B * S,)
        compute_cos_sin_kernel[grid_cos_sin](position_ids, inv, cos, sin, B, S, D)

        # 4) Rotate normalized key rows and scatter into key_cache at cache_position[s]
        # Also scatter original value rows into value_cache at the same positions.
        key_out = torch.empty_like(key_cache)  # [B, N_kv, max_pos, D]
        value_out = torch.empty_like(value_cache)  # [B, N_kv, max_pos, D]

        grid_rotate = (B * S, N_kv)
        rotate_and_scatter_rows_kernel[grid_rotate](
            key_norm, cos, sin,
            key_out, value, value_out,
            cache_position,
            B, N_kv, S, D, D_half, key_cache.shape[2],
            BLOCK_SIZE=D  # single pass over D=128
        )

        # Return: (None for query rotation), updated key_cache, updated value_cache
        return None, key_out, value_out


def run(*args):
    return ModelNew()(*args)
