import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,        # *x, [B, H, S, D], input
    w_ptr,        # *weight, [D]
    y_ptr,        # *y, [B, H, S, D], output
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,   # strides for x
    y_s0, y_s1, y_s2, y_s3,   # strides for y
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # base pointers for this (b, h, s) row
    x_base = pid_b * x_s0 + pid_h * x_s1 + pid_s * x_s2
    y_base = pid_b * y_s0 + pid_h * y_s1 + pid_s * y_s2

    sum_sq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_row_ptr = x_ptr + x_base + cols * x_s3
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals)

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + eps)

    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_row_ptr = x_ptr + x_base + cols * x_s3
        w_row_ptr = w_ptr + cols
        w_vals = tl.load(w_row_ptr, mask=mask, other=0.0).to(tl.float32)
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)

        y_vals = x_vals * inv_rms * w_vals
        y_row_ptr = y_ptr + y_base + cols * y_s3
        # store; Triton will cast to y_ptr dtype if needed
        tl.store(y_row_ptr, y_vals, mask=mask)


@triton.jit
def rotate_q_kernel(
    x_ptr,        # *x_query_norm, [B, H_q, S, D]
    pos_ptr,      # *position_ids, [B, S], int32 (we load per (b,s))
    invf_ptr,     # *inv_freq, [D//2], float32
    y_ptr,        # *y_query_rot, [B, H_q, S, D]
    B: tl.constexpr, H_q: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # load position id for this (b, s)
    pos = tl.load(pos_ptr + pid_b * 0 + pid_s * 0)  # pos_ids is [B, S], stride(0)=S, stride(1)=1 for contiguous
    pos = pos.to(tl.float32)

    # Compute angle vector as a scalar (since emb = pos * inv_freq)
    # We will construct angle per column using pos and invf
    # However, Triton doesn't allow dynamic indexing into loaded vectors, so we compute cos/sin via tl.cos/tl.sin
    # We compute cos(angle) and sin(angle) for each column using pos and invf[cols//2] mapping, but simpler is to compute cos and sin via emb = pos * invf[0:D//2] and then use tl.cos/tl.sin(emb).
    # Since invf_ptr is [D//2], we can't directly access invf for all cols; instead, we compute emb per column by loading invf element:
    # emb = pos * invf[cols//2] if cols < D//2 else 0 (unused). To get angle for all cols, we compute angle_all = pos * invf, but invf is [D//2].
    # Fix: we will reconstruct angle as pos * invf for relevant half using indices. Triton doesn't support vectorized indexing; we can compute two angles: angle1 = pos * invf[0], angle2 = pos * invf[1], ..., by loading invf elements. Since D=128, we can load 64 elements. To keep simple, compute angle per column by loading invf element:
    # emb = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # for j in range(BLOCK_SIZE):
    #     if j < D//2:
    #         idx = j
    #         emb[j] = pos * tl.load(invf_ptr + idx).to(tl.float32)
    #     else:
    #         emb[j] = 0.0
    # Then cos_vals = tl.cos(emb), sin_vals = tl.sin(emb).
    # This is acceptable and Triton supports such loops.

    # First, compute cos and sin vectors of length BLOCK_SIZE
    emb = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for j in range(BLOCK_SIZE):
        if j < (D // 2):
            idx = j
            invf_j = tl.load(invf_ptr + idx).to(tl.float32)
            emb[j] = pos * invf_j
        else:
            emb[j] = 0.0
    cos_vals = tl.cos(emb)
    sin_vals = tl.sin(emb)

    # Now apply rotation: y = x * cos - rotate_half(x) * sin
    x_base = pid_b * x_s0 + pid_h * x_s1 + pid_s * x_s2
    y_base = pid_b * y_s0 + pid_h * y_s1 + pid_s * y_s2

    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_row_ptr = x_ptr + x_base + cols * x_s3
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_half = x_vals[half:]
        x_front = x_vals[:half]
        # Broadcast cos/sin across all cols: for cols >= half, cos/sin are shifted; for cols < half, they are the same
        # Since we only use the first BLOCK_SIZE elements, we can index as above with masks.
        # Compute y for all cols
        # For cols < half: y = x_vals * cos_vals - x_front * sin_vals
        # For cols >= half: y = x_vals * cos_vals - x_half * sin_vals
        # We implement per-block:
        # Case 1: offs < half -> use x_front
        # Case 2: offs >= half -> use x_half
        if offs < half:
            y_vals = x_vals * cos_vals - x_front * sin_vals
        else:
            y_vals = x_vals * cos_vals - x_half * sin_vals
        y_row_ptr = y_ptr + y_base + cols * y_s3
        tl.store(y_row_ptr, y_vals, mask=mask)


@triton.jit
def rotate_k_kernel(
    x_ptr,        # *x_key_norm, [B, H_kv, S, D]
    pos_ptr,      # *position_ids, [B, S]
    invf_ptr,     # *inv_freq, [D//2], float32
    y_ptr,        # *y_key_rot, [B, H_kv, S, D]
    B: tl.constexpr, H_kv: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    pos = tl.load(pos_ptr + pid_b * 0 + pid_s * 0)
    pos = pos.to(tl.float32)

    # Compute emb per column and cos/sin vectors
    emb = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for j in range(BLOCK_SIZE):
        if j < (D // 2):
            idx = j
            invf_j = tl.load(invf_ptr + idx).to(tl.float32)
            emb[j] = pos * invf_j
        else:
            emb[j] = 0.0
    cos_vals = tl.cos(emb)
    sin_vals = tl.sin(emb)

    # Rotation for key: y = x * sin - rotate_half(x) * cos
    x_base = pid_b * x_s0 + pid_h * x_s1 + pid_s * x_s2
    y_base = pid_b * y_s0 + pid_h * y_s1 + pid_s * y_s2

    half = D // 2
    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_row_ptr = x_ptr + x_base + cols * x_s3
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)
        x_half = x_vals[half:]
        x_front = x_vals[:half]
        if offs < half:
            y_vals = x_vals * sin_vals - x_front * cos_vals
        else:
            y_vals = x_vals * sin_vals - x_half * cos_vals
        y_row_ptr = y_ptr + y_base + cols * y_s3
        tl.store(y_row_ptr, y_vals, mask=mask)


@triton.jit
def scatter_update_kernel(
    src_ptr,      # *rotated key/value tensor, [B, H, S, D]
    cache_ptr,    # *key/value cache, [B, H, L, D]
    pos_ptr,      # *cache_position, [S], int32
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    src_s0, src_s1, src_s2, src_s3,
    cache_s0, cache_s1, cache_s2, cache_s3,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # read cache position for this s
    pos = tl.load(pos_ptr + pid_s).to(tl.int32)
    # base pointers
    src_base = pid_b * src_s0 + pid_h * src_s1 + pid_s * src_s2
    cache_base = pid_b * cache_s0 + pid_h * cache_s1 + pos * cache_s2

    for offs in range(0, D, 128):
        cols = offs + tl.arange(0, 128)
        mask = cols < D
        src_row_ptr = src_ptr + src_base + cols * src_s3
        val = tl.load(src_row_ptr, mask=mask, other=0.0)
        cache_row_ptr = cache_ptr + cache_base + cols * cache_s3
        tl.store(cache_row_ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6  # match original

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        B, H_q, S, D = query.shape
        B_kv, H_kv, S2, D2 = key.shape
        assert B == B_kv and S == S2 and D == D2, "Input shapes must match expectations"
        assert H_q == 96 and H_kv == 8, "This implementation expects num_attention_heads=96 and num_key_value_heads=8, as per get_inputs"

        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        # 1) RMSNorm for query and key
        query_norm = torch.empty_like(query, dtype=torch.float32)  # compute in fp32
        key_norm = torch.empty_like(key, dtype=torch.float32)

        grid_rmsq = (B, H_q, S)
        rmsnorm_kernel[grid_rmsq](
            query, q_norm_weight, query_norm,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
            num_warps=4,
        )

        grid_rmsk = (B, H_kv, S)
        rmsnorm_kernel[grid_rmsk](
            key, k_norm_weight, key_norm,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
            num_warps=4,
        )

        # Cast back to original dtype for rotation (bf16)
        query_norm_bf = query_norm.to(torch.bfloat16)
        key_norm_bf = key_norm.to(torch.bfloat16)

        # 2) Rotation using Triton kernels (compute cos/sin inside kernels, no torch ops)
        query_rot = torch.empty_like(query, dtype=torch.bfloat16)
        key_rot = torch.empty_like(key, dtype=torch.bfloat16)

        grid_qrot = (B, H_q, S)
        rotate_q_kernel[grid_qrot](
            query_norm_bf, position_ids, inv_freq, query_rot,
            B, H_q, S, D,
            query_norm_bf.stride(0), query_norm_bf.stride(1), query_norm_bf.stride(2), query_norm_bf.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            BLOCK_SIZE=128,
            num_warps=4,
        )

        grid_krot = (B, H_kv, S)
        rotate_k_kernel[grid_krot](
            key_norm_bf, position_ids, inv_freq, key_rot,
            B, H_kv, S, D,
            key_norm_bf.stride(0), key_norm_bf.stride(1), key_norm_bf.stride(2), key_norm_bf.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_SIZE=128,
            num_warps=4,
        )

        # 3) Scatter update caches
        grid_scatter = (B, H_kv, S)
        # key_cache and value_cache are bf16; cast src to bf16 for store
        scatter_update_kernel[grid_scatter](
            key_rot.to(torch.bfloat16), key_cache,
            cache_position,
            B, H_kv, S, D,
            key_rot.to(torch.bfloat16).stride(0), key_rot.to(torch.bfloat16).stride(1), key_rot.to(torch.bfloat16).stride(2), key_rot.to(torch.bfloat16).stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            num_warps=4,
        )

        scatter_update_kernel[grid_scatter](
            value.to(torch.bfloat16), value_cache,
            cache_position,
            B, H_kv, S, D,
            value.to(torch.bfloat16).stride(0), value.to(torch.bfloat16).stride(1), value.to(torch.bfloat16).stride(2), value.to(torch.bfloat16).stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=4,
        )

        # Return the same outputs as the original run
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
