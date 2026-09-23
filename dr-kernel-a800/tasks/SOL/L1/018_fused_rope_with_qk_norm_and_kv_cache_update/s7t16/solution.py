import torch
import triton
import triton.language as tl

# Kernel: RMSNorm along last dim for a [B, H, S, D] tensor (per row across D)
@triton.jit
def rmsnorm_row_kernel(
    src_ptr,                 # *fp32 or *bf16, input tensor
    dst_ptr,                 # *bf16, output tensor
    weight_ptr,              # *bf16, weight of shape [D]
    B, H, S, D,
    eps,                     # fp32 epsilon
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Base pointers for this (b, h, s) row
    src_base = b * src_stride_b + h * src_stride_h + s * src_stride_s
    dst_base = b * dst_stride_b + h * dst_stride_h + s * dst_stride_s

    # Accumulate sum of squares in fp32
    sumsq = 0.0
    for d in range(0, D):
        x = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.float32)
        sumsq += x * x
    mean = sumsq / D
    inv_rms = tl.math.rsqrt(mean + eps)

    # Normalize and scale by weight, write to dst (bf16)
    for d in range(0, D):
        x = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.float32)
        w = tl.load(weight_ptr + d).to(tl.float32)
        y = x * inv_rms * w
        tl.store(dst_ptr + dst_base + d * dst_stride_d, y.to(tl.bfloat16))


# Kernel: compute cos_all and sin_all per (b, s) into out_cos_ptr/out_sin_ptr [B, S, D]
# We pass pointers to cos/sin outputs and fill them; Triton handles loops over D.
@triton.jit
def rot_vec_kernel(
    out_cos_ptr, out_sin_ptr,
    pos_ptr,                    # *int32, [S], each program_id(2)=s picks pos = pos_ptr[s]
    inv_freq_ptr,               # *fp32, [D//2]
    B, S, D,
    out_cos_stride_b, out_cos_stride_s, out_cos_stride_d,
    out_sin_stride_b, out_sin_stride_s, out_sin_stride_d,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr + s).to(tl.int32)

    # Load angle = pos * inv_freq for d in [0, D//2), then cos/sin for each
    half = D // 2
    for d in range(0, half):
        ang = pos * tl.load(inv_freq_ptr + d)  # fp32
        cos_d = tl.cos(ang)
        sin_d = tl.sin(ang)
        # Store into cos_all and sin_all at [b, s, d] and [b, s, d + half]
        tl.store(out_cos_ptr + b * out_cos_stride_b + s * out_cos_stride_s + d * out_cos_stride_d, cos_d)
        tl.store(out_sin_ptr + b * out_sin_stride_b + s * out_sin_stride_s + d * out_sin_stride_d, sin_d)
    # For d >= half, cos_all[d] = cos(d - half), sin_all[d] = sin(d - half)
    for d in range(half, D):
        idx = d - half
        cos_d = tl.cos(idx)  # previously stored at d
        sin_d = tl.sin(idx)  # previously stored at d
        tl.store(out_cos_ptr + b * out_cos_stride_b + s * out_cos_stride_s + d * out_cos_stride_d, cos_d)
        tl.store(out_sin_ptr + b * out_sin_stride_b + s * out_sin_stride_s + d * out_sin_stride_d, sin_d)


# Kernel: rotate [B, H, S, D] using provided cos_all/sin_all vectors of shape [S, D]
@triton.jit
def rotation_kernel(
    src_ptr, dst_ptr, cos_ptr, sin_ptr,
    B, H, S, D,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_s, dst_stride_d,
    cos_stride_s, cos_stride_d,           # cos_ptr has shape [S, D]
    sin_stride_s, sin_stride_d,           # sin_ptr has shape [S, D]
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    src_base = b * src_stride_b + h * src_stride_h + s * src_stride_s
    dst_base = b * dst_stride_b + h * dst_stride_h + s * dst_stride_s

    for d in range(0, D):
        x = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.float32)

        # Load cos_all and sin_all for this position s
        cos_d = tl.load(cos_ptr + s * cos_stride_s + d * cos_stride_d).to(tl.float32)
        sin_d = tl.load(sin_ptr + s * sin_stride_s + d * sin_stride_d).to(tl.float32)

        x1 = x[..., :D // 2]
        x2 = x[..., D // 2:]

        y = x * cos_d - tl.cat([-x2, x1], axis=-1) * sin_d
        tl.store(dst_ptr + dst_base + d * dst_stride_d, y.to(tl.bfloat16))


# Kernel: scatter rotated keys/values into key_cache/value_cache at cache_position[s]
@triton.jit
def scatter_update_cache_kernel(
    src_ptr,                   # *bf16 or *fp32, [B, H, D]
    key_ptr,                   # *bf16, [B, H, L, D]
    value_ptr,                # *bf16, [B, H, D] or *bf16, same as src
    cache_pos_ptr,            # *int32, [S]
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_d,
    key_stride_b, key_stride_h, key_stride_l, key_stride_d,
    value_stride_b, value_stride_h, value_stride_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    src_base = b * src_stride_b + h * src_stride_h
    key_base = b * key_stride_b + h * key_stride_h + pos * key_stride_l
    val_base = b * value_stride_b + h * value_stride_h

    for d in range(0, D):
        src_val = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.bfloat16)
        tl.store(key_ptr + key_base + d * key_stride_d, src_val)
        # Store value into value_cache at same [b, h, pos, :]
        val = tl.load(value_ptr + val_base + d * value_stride_d).to(tl.bfloat16)
        tl.store(value_ptr + key_base + d * key_stride_d, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        position_ids = position_ids.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()

        # Shapes
        B_q, H_q, S, D = query.shape  # H_q can be 96
        B_k, H_k, S, D = key.shape    # H_k can be 8
        B_bc, H_bc, L, D_cache = key_cache.shape  # L == 262144, D_cache == D=128

        # 1) RMSNorm for query and key (compute in fp32, return in bf16)
        query_norm = torch.empty_like(query, dtype=torch.bfloat16)
        key_norm = torch.empty_like(key, dtype=torch.bfloat16)

        grid_q = (B_q, H_q, S)
        rmsnorm_row_kernel[grid_q](
            query, query_norm, q_norm_weight, rms_norm_eps,
            B_q, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            num_warps=4, num_stages=2,
        )

        grid_k = (B_k, H_k, S)
        rmsnorm_row_kernel[grid_k](
            key, key_norm, k_norm_weight, rms_norm_eps,
            B_k, H_k, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            num_warps=4, num_stages=2,
        )

        # 2) Compute cos_all and sin_all per (b, s) using Triton
        # Allocate outputs: cos_all, sin_all of shape [B, S, D] in fp32
        cos_all = torch.empty((B_q, S, D), device=query.device, dtype=torch.float32)
        sin_all = torch.empty((B_q, S, D), device=query.device, dtype=torch.float32)

        grid_rot = (B_q, S)
        rot_vec_kernel[grid_rot](
            cos_all, sin_all,
            cache_position, inv_freq,
            B_q, S, D,
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            num_warps=2, num_stages=2,
        )

        # 3) Rotate query_norm and key_norm
        query_rot = torch.empty_like(query_norm, dtype=torch.bfloat16)
        key_rot = torch.empty_like(key_norm, dtype=torch.bfloat16)

        grid_qrot = (B_q, H_q, S)
        rotation_kernel[grid_qrot](
            query_norm, query_rot, cos_all, sin_all,
            B_q, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            cos_all.stride(1), cos_all.stride(2),  # cos_all has shape [S, D]
            sin_all.stride(1), sin_all.stride(2),  # sin_all has shape [S, D]
            num_warps=4, num_stages=2,
        )

        grid_krot = (B_k, H_k, S)
        rotation_kernel[grid_krot](
            key_norm, key_rot, sin_all, cos_all,
            B_k, H_k, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(1), sin_all.stride(2),
            num_warps=4, num_stages=2,
        )

        # 4) Scatter update caches: write rotated keys and original values at cache_position
        L = key_cache.shape[2]
        grid_scatter = (B_k, H_k, S)
        scatter_update_cache_kernel[grid_scatter](
            key_rot, key_cache,
            key_rot,  # value to scatter (we use rotated key here as per original run; if original uses rotated key, this matches)
            cache_position,
            B_k, H_k, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            None,  # no value_stride argument needed for rotated keys
            num_warps=4, num_stages=2,
        )

        # If we need to write original values (as in original run: value_cache[:, :, cache_position] = value),
        # we can reuse a similar scatter kernel with value (here we use rotated key as original code does).
        # To strictly match original behavior for value_cache update, we can allocate another rotated value tensor and scatter it.
        # However, original code assigns original 'value' (not rotated) to cache; since we need Triton-only,
        # we can still scatter original 'value' using another kernel launch (we defined and will use it here):
        # But scatter_update_cache_kernel expects src of same [B, H, D]; we can pass value directly as src (bf16).
        # Note: the original code assigns original 'value' (not rotated). We'll do that here as well.

        # Prepare rotated_value (not used in original, but we keep consistency): create rotated_value = value (original run used value directly)
        # However, original run uses value directly (not rotated), so we should not rotate value. For cache update, we should scatter original value.
        # But ModelNew.forward must use Triton; we can pass value as src_ptr and store into value_cache at cache_position.

        # We need a separate scatter for original values (not rotated). Let's implement it by copying the logic: src_ptr = value, dst_ptr = value_cache.

        # 4b) Scatter original values into value_cache at cache_position
        # First, allocate rotated_value tensor identical to value (bf16)
        rotated_value = value.contiguous()  # original code does not rotate values, so rotated_value = value
        # Launch scatter_update_cache_kernel with src_ptr = rotated_value (which is value), and dst_ptr = value_cache, cache_pos_ptr = cache_position

        # Note: In the original run, value is not rotated; we can reuse the same kernel by setting src_ptr to value and dst_ptr to value_cache.
        # But since we only have one kernel, we need to pass the correct pointers. We'll pass value as src and value_cache as dst.

        grid_scatter_val = (B_k, H_k, S)
        scatter_update_cache_kernel[grid_scatter_val](
            value, value_cache,
            value,  # dummy second argument not used
            cache_position,
            B_k, H_k, S, D, L,
            value.stride(0), value.stride(1), value.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            None,  # no value_stride argument needed
            num_warps=4, num_stages=2,
        )

        # Return rotated query, rotated key, updated key_cache, updated value_cache
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
