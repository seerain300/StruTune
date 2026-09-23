import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm along the last dimension (D) for a 4D tensor [B, H, S, D]
# Input x: fp16/bf16, weight: fp32 [D], output y: same dtype as x
@triton.jit
def rmsnorm_4d_kernel(
    x_ptr, y_ptr, weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Accumulate sum of squares across D in fp32
    sumsq = 0.0
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and apply weight, write back
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        w = tl.load(weight_ptr + d, mask=mask, other=1.0).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        # Store to original dtype inferred from pointer type (bf16 here)
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + d * y_stride_d, y, mask=mask)
        offs += BLOCK_SIZE

# Triton kernel: apply rotation to a 4D tensor [B, H, S, D] using provided cos_all/sin_all (length 2*D vectors)
# For query: y = x * cos - rotate_half(x) * sin
# For key: y = x * sin - rotate_half(x) * cos  (using sin_all as given in original logic)
@triton.jit
def rotate_4d_kernel(
    x_ptr, y_ptr, cos_ptr, sin_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load whole row x across D
    offs = 0
    x_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        x_row = tl.where(mask, x.to(tl.float32), x_row)
        offs += BLOCK_SIZE

    # Build rotated components: rotate_half(x) = [-x[..., D:], x[..., :D]]
    half = D // 2
    r_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    r_x[:half] = -x_row[half:]
    r_x[half:] = x_row[:half]

    # Load cos/sin vectors (length 2*D)
    d = tl.arange(0, BLOCK_SIZE)
    cos_vec = tl.load(cos_ptr + d, mask=d < 2 * D, other=1.0).to(tl.float32)
    sin_vec = tl.load(sin_ptr + d, mask=d < 2 * D, other=0.0).to(tl.float32)

    # Compose y depending on kernel usage: query vs key
    # Here we implement both modes by flipping signs based on a constexpr flag. For query, mode=0 uses y = x*cos - r_x*sin
    # For key, mode=1 uses y = x*sin - r_x*cos. We pass mode via grid launch parameter 'mode' (not used here; we set mode=0 for query and mode=1 for key).
    mode = 0  # must be set at launch time, but Triton does not support passing extra flags; define two versions below.

# We will define two rotation kernels: query and key, with different mode logic.
@triton.jit
def rotate_query_4d_kernel(
    x_ptr, y_ptr, cos_ptr, sin_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    offs = 0
    x_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        x_row = tl.where(mask, x.to(tl.float32), x_row)
        offs += BLOCK_SIZE

    half = D // 2
    r_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    r_x[:half] = -x_row[half:]
    r_x[half:] = x_row[:half]

    d = tl.arange(0, BLOCK_SIZE)
    cos_vec = tl.load(cos_ptr + d, mask=d < 2 * D, other=1.0).to(tl.float32)
    sin_vec = tl.load(sin_ptr + d, mask=d < 2 * D, other=0.0).to(tl.float32)

    y_row = x_row * cos_vec - r_x * sin_vec

    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + d * y_stride_d, y_row, mask=mask)
        offs += BLOCK_SIZE

@triton.jit
def rotate_key_4d_kernel(
    x_ptr, y_ptr, sin_ptr, cos_ptr,  # pass sin first, cos second (even though cos_ptr is unused in computation, sin is used)
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    offs = 0
    x_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + d * x_stride_d, mask=mask, other=0.0)
        x_row = tl.where(mask, x.to(tl.float32), x_row)
        offs += BLOCK_SIZE

    half = D // 2
    r_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    r_x[:half] = -x_row[half:]
    r_x[half:] = x_row[:half]

    d = tl.arange(0, BLOCK_SIZE)
    sin_vec = tl.load(sin_ptr + d, mask=d < 2 * D, other=0.0).to(tl.float32)
    cos_vec = tl.load(cos_ptr + d, mask=d < 2 * D, other=1.0).to(tl.float32)

    # Note: original logic uses sin_all for key rotation. We implement y = x * sin - r_x * cos.
    # Since cos_all is not used for key rotation in original, we still load cos_ptr to satisfy signature, but it won't be used in computation.
    y_row = x_row * sin_vec - r_x * cos_vec

    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + d * y_stride_d, y_row, mask=mask)
        offs += BLOCK_SIZE

# Triton kernel: scatter update key/value caches at cache_position indices
# y_src: [B, H, S, D], key_cache: [B, H, L, D]
# For each (b,h,s), write y_src[b,h,s,:] into key_cache[b,h, cache_pos[s], :]
@triton.jit
def scatter_cache_update_kernel(
    src_ptr, dst_ptr,
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(cache_pos_ptr + s).to(tl.int32)
    # Load entire row x
    offs = 0
    x_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        x = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + d * src_stride_d, mask=mask, other=0.0)
        x_row = tl.where(mask, x.to(tl.float32), x_row)
        offs += BLOCK_SIZE

    # Store to cache at position pos
    offs = 0
    while offs < D:
        d = offs + tl.arange(0, BLOCK_SIZE)
        mask = d < D
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l + d * dst_stride_d, x_row, mask=mask)
        offs += BLOCK_SIZE

def _launch_rmsnorm(query, weight_q, eps, device):
    B_q, H_q, S, D = query.shape
    query_norm = torch.empty_like(query)
    grid = (B_q, H_q, S)
    rmsnorm_4d_kernel[grid](
        query, query_norm, weight_q.to(torch.float32),
        B_q, H_q, S, D,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        eps,
        BLOCK_SIZE=128, num_warps=4,
    )
    return query_norm

def _launch_rmsnorm_key(key, weight_k, eps, device):
    B_q, H_k, S, D = key.shape
    key_norm = torch.empty_like(key)
    grid = (B_q, H_k, S)
    rmsnorm_4d_kernel[grid](
        key, key_norm, weight_k.to(torch.float32),
        B_q, H_k, S, D,
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        eps,
        BLOCK_SIZE=128, num_warps=4,
    )
    return key_norm

def _launch_rotation_query(query_norm, cos_all, sin_all, device):
    B_q, H_q, S, D = query_norm.shape
    query_rot = torch.empty_like(query_norm)
    grid = (B_q, H_q, S)
    rotate_query_4d_kernel[grid](
        query_norm.to(torch.float32), query_rot, cos_all.to(torch.float32), sin_all.to(torch.float32),
        B_q, H_q, S, D,
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
        BLOCK_SIZE=128, num_warps=4,
    )
    return query_rot

def _launch_rotation_key(key_norm, sin_all, cos_all, device):
    B_q, H_k, S, D = key_norm.shape
    key_rot = torch.empty_like(key_norm)
    grid = (B_q, H_k, S)
    rotate_key_4d_kernel[grid](
        key_norm.to(torch.float32), key_rot, sin_all.to(torch.float32), cos_all.to(torch.float32),
        B_q, H_k, S, D,
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        key_rot.stride(0), key_norm.stride(1), key_norm.stride(2), key_rot.stride(3),  # key_rot.stride(2) is d, but we pass key_norm.stride(1) mistakenly; fix below
        BLOCK_SIZE=128, num_warps=4,
    )
    # Correct grid for key rotation should use key_rot's strides for (b,h,l,d); fix:
    # We need dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d for key_cache and key_rot. Pass key_rot strides:
    return key_rot

# Note: The above has a bug in stride usage; we need to pass correct strides. Fix below:
def _launch_rotation_key(key_norm, sin_all, cos_all, device):
    B_q, H_k, S, D = key_norm.shape
    key_rot = torch.empty_like(key_norm)
    grid = (B_q, H_k, S)
    # We'll pass key_rot strides correctly:
    rotate_key_4d_kernel[grid](
        key_norm.to(torch.float32), key_rot, sin_all.to(torch.float32), cos_all.to(torch.float32),
        B_q, H_k, S, D,
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        BLOCK_SIZE=128, num_warps=4,
    )
    return key_rot

def _launch_scatter_update(key_rot, value, key_cache, value_cache, cache_position, device):
    B_q, H_k, S, D = key_rot.shape
    # We need to store into key_cache and value_cache at indices cache_position[s] per (b,h,s)
    # For key_cache: write key_rot[b,h,s,:] at cache_position[s]
    # For value_cache: write value[b,h,s,:] at cache_position[s]
    # We'll handle key_cache first:
    grid = (B_q, H_k, S)
    scatter_cache_update_kernel[grid](
        key_rot.to(torch.float32), key_cache,  # src and dst pointers
        B_q, H_k, S, D, key_cache.shape[2],
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        cache_position.to(torch.int32),
        BLOCK_SIZE=128, num_warps=4,
    )
    # For value_cache: similarly
    grid = (B_q, H_k, S)
    scatter_cache_update_kernel[grid](
        value.to(torch.float32), value_cache,
        B_q, H_k, S, D, value_cache.shape[2],
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        cache_position.to(torch.int32),
        BLOCK_SIZE=128, num_warps=4,
    )

class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure device/dtype
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be CUDA tensors."
        assert query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16, "Expected bf16 inputs."
        assert key_cache.shape == (query.shape[0], key.shape[1], key_cache.shape[2], query.shape[3]) and value_cache.shape == (query.shape[0], key.shape[1], value_cache.shape[2], query.shape[3]), "Cache shapes must match expected [B, num_kv_heads, L, D]."
        # 1) RMSNorm
        query_norm = _launch_rmsnorm(query, q_norm_weight, rms_norm_eps, query.device)
        key_norm = _launch_rmsnorm_key(key, k_norm_weight, rms_norm_eps, key.device)
        # 2) Compute rotation vectors on device using torch (since Triton cannot access torch tensors inside kernels, we precompute per-position cos_all/sin_all)
        # For each (b, s), pos = position_ids[b, s]; angle = pos * inv_freq (length D//2)
        B_q, H_q, S, D = query.shape
        pos_vec = position_ids.reshape(-1).to(torch.int32)  # [B_q*S]
        # Build cos_all and sin_all of length 2*D:
        # query uses cos_all = [cos(angle), cos(angle)], sin_all = [sin(angle), sin(angle)]
        # key uses sin_all = [sin(angle), sin(angle)]
        angles = (pos_vec.to(torch.float32) * inv_freq.to(torch.float32)).unsqueeze(1)  # [P, 64]
        # cos_all, sin_all: [2*D]
        cos_all = torch.cat([torch.cos(angles), torch.cos(angles)], dim=1).reshape(-1)  # [2*D]
        sin_all = torch.cat([torch.sin(angles), torch.sin(angles)], dim=1).reshape(-1)  # [2*D]
        # 3) Apply rotation
        query_rot = _launch_rotation_query(query_norm, cos_all, sin_all, query.device)
        key_rot = _launch_rotation_key(key_norm, sin_all, cos_all, key.device)  # note: sin_all for key; cos_all not used in original but we pass it to satisfy kernel signature
        # 4) Scatter update caches
        _launch_scatter_update(key_rot, value, key_cache, value_cache, cache_position, key.device)
        # Return rotated query and key and updated caches (the evaluation harness expects returns from run; this forward returns None for caches)
        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
