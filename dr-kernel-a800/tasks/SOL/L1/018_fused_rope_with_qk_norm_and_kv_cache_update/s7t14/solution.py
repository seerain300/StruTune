import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm per [b, h, s] row along last dim D=128
# y = x * (weight / sqrt(mean(x^2) + eps))
@triton.jit
def rmsnorm_row_kernel(
    x_ptr,        # *ptr to input [B, H, S, D]
    y_ptr,        # *ptr to output [B, H, S, D]
    w_ptr,        # *ptr to weight [D] (same for all rows)
    eps,          # epsilon float
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # base pointer for this row
    base = b * x_stride_b + h * x_stride_h + s * x_stride_s

    # Accumulate sum of squares in fp32
    sum_sq = 0.0
    for d in range(0, D):
        x_val = tl.load(x_ptr + base + d * x_stride_d)
        x_f32 = x_val.to(tl.float32)
        sum_sq += x_f32 * x_f32

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Scale by weight and store
    for d in range(0, D):
        x_val = tl.load(x_ptr + base + d * x_stride_d)
        x_f32 = x_val.to(tl.float32)
        w_val = tl.load(w_ptr + d).to(tl.float32)
        y_val = x_f32 * inv_rms * w_val
        # y_val is fp32; store as original dtype (bf16) via cast
        tl.store(y_ptr + base + d * y_stride_d, y_val.to(x_val.dtype))


# Triton kernel: apply rotation to a tensor [B, H, S, D]
# y = x * cos - rotate_half(x) * sin
# cos, sin: [D] already constructed as [cos, cos] or [sin, sin]
@triton.jit
def rotate_half_sin_kernel(
    x_ptr,        # *ptr to input [B, H, S, D] (normalized)
    y_ptr,        # *ptr to output [B, H, S, D]
    cos_ptr,      # *ptr to cos vector [D]
    sin_ptr,      # *ptr to sin vector [D]
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base = b * x_stride_b + h * x_stride_h + s * x_stride_s

    # First half [0:D//2]
    for d in range(0, D // 2):
        x1 = tl.load(x_ptr + base + d * x_stride_d).to(tl.float32)
        cos1 = tl.load(cos_ptr + d).to(tl.float32)
        sin1 = tl.load(sin_ptr + d).to(tl.float32)
        # rotate_half(x) for first half is x1 (no negation)
        y1 = x1 * cos1 - x1 * sin1  # keys use sin, cos mixed here but sin_all_k is cos_all, cos_all_k is sin_all; see caller
        tl.store(y_ptr + base + d * y_stride_d, y1.to(tl.float32))  # store fp32; will cast outside if needed

    # Second half [D//2:D]
    for d in range(0, D // 2):
        x2 = tl.load(x_ptr + base + (d + D // 2) * x_stride_d).to(tl.float32)
        cos2 = tl.load(cos_ptr + (d + D // 2)).to(tl.float32)
        sin2 = tl.load(sin_ptr + (d + D // 2)).to(tl.float32)
        # rotate_half(x) for second half is -x2
        y2 = x2 * cos2 - (-x2) * sin2
        tl.store(y_ptr + base + (d + D // 2) * y_stride_d, y2.to(tl.float32))


# Triton kernel: scatter write into key_cache at positions cache_pos[s]
# Input: key_rot [B, H, S, D], cache_pos [S], key_cache [B, H, L, D]
@triton.jit
def scatter_update_cache_kernel(
    src_ptr,      # *ptr to rotated keys [B, H, S, D], contiguous
    dst_ptr,      # *ptr to key_cache [B, H, L, D]
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # *ptr to int32 cache positions [S]
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # load cache position
    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    # base pointers for src and dst rows
    src_base = b * src_stride_b + h * src_stride_h + s * src_stride_s
    dst_base = b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l

    for d in range(0, D):
        src_val = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.float32)
        tl.store(dst_ptr + dst_base + d * dst_stride_d, src_val.to(tl.float32))


# Triton kernel: scatter write into value_cache at positions cache_pos[s]
# Input: value [B, H, S, D], cache_pos [S], value_cache [B, H, L, D]
@triton.jit
def scatter_update_value_cache_kernel(
    src_ptr,      # *ptr to value [B, H, S, D]
    dst_ptr,      # *ptr to value_cache [B, H, L, D]
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # *ptr to int32 cache positions [S]
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    src_base = b * src_stride_b + h * src_stride_h + s * src_stride_s
    dst_base = b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l

    for d in range(0, D):
        src_val = tl.load(src_ptr + src_base + d * src_stride_d).to(tl.float32)
        tl.store(dst_ptr + dst_base + d * dst_stride_d, src_val.to(tl.float32))


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

        B_q, H_q, S, D = query.shape  # H_q can be 96
        B_k, H_k, S, D = key.shape    # H_k can be 8
        B_bc, H_bc, L, D_cache = key_cache.shape  # L == 262144, D_cache == D=128

        # 1) RMSNorm for query and key (compute in fp32)
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch RMSNorm for query
        grid_q = (B_q, H_q, S)
        rmsnorm_row_kernel[grid_q](
            query, query_norm, q_norm_weight, rms_norm_eps,
            B_q, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query.norm.stride(3),
        )

        # Launch RMSNorm for key
        grid_k = (B_k, H_k, S)
        rmsnorm_row_kernel[grid_k](
            key, key_norm, k_norm_weight, rms_norm_eps,
            B_k, H_k, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        )

        # Generate cos_all and sin_all using torch (host) for perfect parity with original:
        # angle = pos * inv_freq, inv_freq has length D//2=64
        # cos_all_q = [cos(angle), cos(angle)] and sin_all_q = [sin(angle), sin(angle)]
        # For keys, we use sin_all_q (cos_all_k) and cos_all_q (sin_all_k) as per original comment and PyTorch code behavior.
        # Note: position_ids is [B, S], we need to expand over H dimensions; but original code applies per [b, s].
        # We build per [B, S] and then use in kernels. Triton kernels take these vectors, no dynamic cat inside.
        # Prepare angles
        # Ensure inv_freq is float32
        inv_freq = inv_freq.to(torch.float32).to(query.device)
        # position_ids -> [B, S]
        # angle = position_ids (int) * inv_freq (float)
        pos = position_ids  # [B, S], int64
        angle = (pos.to(torch.float32) * inv_freq).view(B_k, S, D // 2)  # [B, S, 64]
        cos_half = torch.cos(angle)  # [B, S, 64]
        sin_half = torch.sin(angle)  # [B, S, 64]

        # Concatenate to length D along last dim
        # cos_all_q = [cos_half, cos_half] -> [B, S, D]
        # sin_all_q = [sin_half, sin_half]
        cos_all_q = torch.cat([cos_half, cos_half], dim=-1)  # [B, S, D]
        sin_all_q = torch.cat([sin_half, sin_half], dim=-1)  # [B, S, D]

        # For keys, we use sin_all_k = cos_all_q (cos-based rotation per original code), and cos_all_k = sin_all_q
        sin_all_k = cos_all_q  # [B, S, D]
        cos_all_k = sin_all_q  # [B, S, D]

        # 2) Apply rotation in Triton: query and key
        # query rotation: y = x * cos_all_q - rotate_half(x) * sin_all_q
        query_rot = torch.empty_like(query_norm, dtype=torch.float32, device=query.device)

        grid_qrot = (B_q, H_q, S)
        rotate_half_sin_kernel[grid_qrot](
            query_norm, query_rot, cos_all_q, sin_all_q,
            B_q, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
        )

        # key rotation: y = x * sin_all_k - rotate_half(x) * cos_all_k
        key_rot = torch.empty_like(key_norm, dtype=torch.float32, device=key.device)

        grid_krot = (B_k, H_k, S)
        rotate_half_sin_kernel[grid_krot](
            key_norm, key_rot, sin_all_k, cos_all_k,
            B_k, H_k, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        )

        # Cast back to original dtype (bf16) for output tensors (not used further, but matches original interface)
        query_rot = query_rot.to(torch.bfloat16)
        key_rot = key_rot.to(torch.bfloat16)

        # 3) Scatter update caches: write rotated keys and original values at cache_position
        # Cast sources to bf16 for store
        key_rot_bf16 = key_rot.to(torch.bfloat16)
        value_bf16 = value.to(torch.bfloat16)

        # int32 cache positions
        cache_pos_i32 = cache_position.to(torch.int32)

        grid_scatter = (B_k, H_k, S)
        scatter_update_cache_kernel[grid_scatter](
            key_rot_bf16, key_cache,
            B_k, H_k, S, D, L,
            key_rot_bf16.stride(0), key_rot_bf16.stride(1), key_rot_bf16.stride(2), key_rot_bf16.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_pos_i32,
        )

        scatter_update_value_cache_kernel[grid_scatter](
            value_bf16, value_cache,
            B_k, H_k, S, D, L,
            value_bf16.stride(0), value_bf16.stride(1), value_bf16.stride(2), value_bf16.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            cache_pos_i32,
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
