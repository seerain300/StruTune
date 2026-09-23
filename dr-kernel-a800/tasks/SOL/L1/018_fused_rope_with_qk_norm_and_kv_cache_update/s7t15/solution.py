import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    src_ptr,                 # *bf16, [B, H, S, D]
    weight_ptr,             # *bf16, [D]
    eps,                    # fp32 scalar
    out_ptr,                # *bf16, [B, H, S, D]
    B, H, S, D,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_src = b * src_stride_b + h * src_stride_h + s * src_stride_s
    base_out = b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Compute sum of squares across D
    sum_sq = 0.0
    d_loop = 0
    while d_loop < D:
        x = tl.load(src_ptr + base_src + d_loop * src_stride_d).to(tl.float32)
        sum_sq += x * x
        d_loop += 1
    mean = sum_sq / D
    inv = 1.0 / tl.sqrt(mean + eps)

    # Normalize and scale by weight
    d_vec = tl.arange(0, D)
    for d in range(0, D):
        x = tl.load(src_ptr + base_src + d * src_stride_d).to(tl.float32)
        w = tl.load(weight_ptr + d).to(tl.float32)
        y = x * inv * w
        tl.store(out_ptr + base_out + d * out_stride_d, y.to(tl.bfloat16))


@triton.jit
def rotation_kernel(
    x_ptr,                   # *bf16, [B, H, S, D]
    cos_ptr, sin_ptr,       # *bf32 or *bf16, [B, H, D] (we pass bf32)
    out_ptr,                # *bf16, [B, H, S, D]
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    cos_stride_b, cos_stride_h, cos_stride_d,
    sin_stride_b, sin_stride_h, sin_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    is_query: tl.constexpr,  # 1 for query, 0 for key
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride_b + h * x_stride_h + s * x_stride_s
    base_out = b * out_stride_b + h * out_stride_h + s * out_stride_s

    d_vec = tl.arange(0, D)
    x_vec = tl.load(x_ptr + base_x + d_vec * x_stride_d).to(tl.float32)

    # Load cos/sin vectors for this (b,h)
    cos_vec = tl.load(cos_ptr + b * cos_stride_b + h * cos_stride_h + d_vec * cos_stride_d).to(tl.float32)
    sin_vec = tl.load(sin_ptr + b * sin_stride_b + h * sin_stride_h + d_vec * sin_stride_d).to(tl.float32)

    # rotate_half(x) = [-x[D//2:], x[:D//2]]
    half = D // 2
    x_second = x_vec[half:]
    x_first = x_vec[:half]
    x_rot = tl.cat([-x_second, x_first], axis=0)

    if is_query:
        y = x_vec * cos_vec - x_rot * sin_vec
    else:
        # key uses sin for first half and cos for second half: y = x * sin - rotate_half(x) * cos
        y = x_vec * sin_vec - x_rot * cos_vec

    tl.store(out_ptr + base_out + d_vec * out_stride_d, y.to(tl.bfloat16))


@triton.jit
def scatter_update_cache_kernel(
    src_ptr,                # *bf16 or *fp32, [B, H, D] depending on input
    key_ptr,                # *bf16, [B, H, L, D]
    value_ptr,              # *bf16, [B, H, D] or *bf16, same as src
    cache_pos_ptr,          # *int32, [S]
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
        B_bc, H_bc, L, D_cache = key_cache.shape  # L == 262144, D_cache


def run(*args):
    return ModelNew()(*args)
