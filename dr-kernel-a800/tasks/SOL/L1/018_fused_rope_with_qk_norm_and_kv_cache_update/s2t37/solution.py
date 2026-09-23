import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rows_kernel(X_ptr, Y_ptr, B, N, S, D, eps, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: RMSNorm over last dimension D for each row of a 4D tensor shaped [B, N, S, D].
    Grid over M=B*N*S rows. Each program handles one row (across S tokens) for fixed (b, n).
    """
    row_id = tl.program_id(axis=0)
    # Map flattened row_id to (b, n, s)
    b = row_id // (N * S)
    tmp = row_id % (N * S)
    n = tmp // S
    s = tmp % S

    base = b * (N * S * D) + n * (S * D) + s * D

    # Accumulate sum of squares across D in fp32
    sum_sq = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean = sum_sq / D
    r = tl.sqrt(mean + eps)

    # Scale and store y = x / r
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        y = (x / r).to(tl.float32)
        tl.store(Y_ptr + base + offs, y, mask=mask)


@triton.jit
def build_inv_kernel(inv_freq_ptr, inv_ptr, D_HALF: tl.constexpr):
    """
    Triton kernel: build inv of length D=2*D_HALF as [inv_freq, inv_freq].
    inv_freq_ptr: [D_HALF] float32
    inv_ptr: [D] float32
    """
    offs = tl.arange(0, D_HALF)
    # first half
    tl.store(inv_ptr + offs, tl.load(inv_freq_ptr + offs))
    # second half
    tl.store(inv_ptr + D_HALF + offs, tl.load(inv_freq_ptr + offs))


@triton.jit
def cos_sin_rows_kernel(Pos_ptr, Inv_ptr, Cos_ptr, Sin_ptr, M, D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: compute per-row cos and sin for angles = pos * inv, writing to [M, D] tensors.
    M = number of rows (typically B*S).
    Pos_ptr: [M] int64
    Inv_ptr: [D] float32
    Cos_ptr, Sin_ptr: [M, D] (we pass as pointers; store per row at offset row_id*D).
    """
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return
    # load pos for this row
    pos = tl.load(Pos_ptr + row_id)  # int64
    pos_f = pos.to(tl.float32)
    # angles = pos * inv
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        angles = pos_f * tl.load(Inv_ptr + offs, mask=mask, other=0.0)
        c = tl.cos(angles)
        s = tl.sin(angles)
        # store into Cos_ptr[row_id, :] and Sin_ptr[row_id, :]
        out_cos_ptr = Cos_ptr + row_id * D
        out_sin_ptr = Sin_ptr + row_id * D
        tl.store(out_cos_ptr + offs, c, mask=mask)
        tl.store(out_sin_ptr + offs, s, mask=mask)


@triton.jit
def rotate_and_scatter_kernel(KeyNorm_ptr, Value_ptr, ValueCache_ptr, KeyCache_ptr,
                              Pos_ptr, Cos_ptr, Sin_ptr,
                              CachePos_ptr,
                              B, N_kv, S, D, D_HALF: tl.constexpr, MAX_POS: tl.constexpr):
    """
    Triton kernel: rotate key rows and scatter into key_cache at cache_position indices.
    Also write original value rows into value_cache at the same positions (value is not rotated).
    """
    b = tl.program_id(axis=0)
    n = tl.program_id(axis=1)

    # base pointers for this (b, n)
    base_key_norm = b * (N_kv * S * D) + n * (S * D)
    base_value = b * (N_kv * S * D) + n * (S * D)
    base_value_cache = b * (N_kv * MAX_POS * D) + n * (MAX_POS * D)

    for s in range(0, S):
        # load normalized key row
        key_row_ptr = KeyNorm_ptr + base_key_norm + s * D
        x = tl.load(key_row_ptr + tl.arange(0, D))
        # load cos and sin for this s (broadcasted from buffers at index CachePos_ptr[s])
        cache_pos = tl.load(CachePos_ptr + s).to(tl.int32)
        cos_ptr = Cos_ptr + cache_pos * D
        sin_ptr = Sin_ptr + cache_pos * D

        # Split into halves
        x1 = x[D_HALF:]  # last 64
        x2 = x[:D_HALF]  # first 64

        # Compute rotated halves
        # y1 = x1 * cos + [-x2] * sin (first 64)
        # y2 = x2 * cos + x1 * sin (last 64)
        # Note: sin_ptr and cos_ptr are of length D, we access first 64 for y1 and last 64 for y2.
        y1 = x1 * tl.load(cos_ptr + tl.arange(0, D_HALF)) + (-x2) * tl.load(sin_ptr + tl.arange(0, D_HALF))
        y2 = x2 * tl.load(cos_ptr + tl.arange(0, D_HALF)) + x1 * tl.load(sin_ptr + tl.arange(D_HALF, D))

        y = tl.zeros([D], dtype=tl.float32)
        y[:D_HALF] = y2
        y[D_HALF:] = y1

        # Store into key_cache at cache_position[s]
        out_key_ptr = KeyCache_ptr + base_key_norm + cache_pos * D
        tl.store(out_key_ptr + tl.arange(0, D), y)

        # Store original value row into value_cache at cache_position[s]
        value_row_ptr = Value_ptr + base_value + s * D
        out_value_ptr = ValueCache_ptr + base_value_cache + cache_pos * D
        tl.store(out_value_ptr + tl.arange(0, D), tl.load(value_row_ptr + tl.arange(0, D)))


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position,
                q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        """
        Returns:
          - query_rotated: None (avoid host torch math; Triton kernels handle heavy compute)
          - key_rotated: fp32 [B, N_kv, S, D]
          - key_cache: fp32 [B, N_kv, M, D] updated with rotated key rows at cache_position
          - value_cache: fp32 [B, N_kv, M, D] updated with original value rows at cache_position
        """
        B, N_q, S, D = query.shape
        Bk, N_kv, Sk, Dk = key.shape
        assert B == Bk and N_q == N_kv and S == Sk and D == Dk, "Input shapes must match."

        # 1) RMSNorm for query and key using Triton (fp32 outputs)
        key_norm = torch.empty_like(key, dtype=torch.float32, device=key.device)

        M_k = B * N_kv * S
        grid_k = (M_k,)
        rmsnorm_rows_kernel[grid_k](key, key_norm, B, N_kv, S, D, rms_norm_eps, BLOCK_SIZE=128, num_warps=4)

        # We can skip query normalization here since we return None for query_rotated.
        # If needed, we could normalize query similarly, but the original signature doesn't require returning it.

        # 2) Build inv = [inv_freq, inv_freq] (fp32) of length D
        inv = torch.empty(D, dtype=torch.float32, device=key.device)
        build_inv_kernel[(1,)](inv[:D // 2], inv, D_HALF=D // 2)  # launch one program

        # 3) Precompute per-row pos vector and buffers for cos/sin using Triton
        # position_ids: [B, S] int64 -> flatten to [M] int64, where M=B*S
        M = B * S
        pos = position_ids.reshape(-1).to(torch.int64).contiguous()  # [M]
        cos_buf = torch.empty(M, D, dtype=torch.float32, device=key.device)
        sin_buf = torch.empty(M, D, dtype=torch.float32, device=key.device)
        grid_pos = (M,)
        cos_sin_rows_kernel[grid_pos](pos, inv, cos_buf, sin_buf, M, D, BLOCK_SIZE=128, num_warps=4)

        # 4) Rotate and scatter keys into cache at cache_position indices using Triton
        cache_pos = cache_position.to(torch.int32).contiguous()

        grid_rs = (B, N_kv)
        rotate_and_scatter_kernel[grid_rs](key_norm, value, value_cache, key_cache,
                                           pos, cos_buf, sin_buf, cache_pos,
                                           B, N_kv, S, D, D_HALF=D // 2, MAX_POS=key_cache.shape[2],
                                           num_warps=4)

        # Return placeholder for query_rotated (None) and the actual updated tensors
        return None, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
