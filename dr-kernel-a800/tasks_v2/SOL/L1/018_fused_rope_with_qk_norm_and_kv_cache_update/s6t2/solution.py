import torch
import triton
import triton.language as tl


# -----------------------
# Triton kernels
# -----------------------

# 1) RMSNorm reduction: compute sum of squares across head_dim for each (b, h, s)
@triton.jit
def rms_sum_kernel(query_ptr, sum_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,  # head_dim, e.g., 128
                    batch_stride, qh_stride, s_stride, d_stride):
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    h = rem // S
    s = rem % S

    total_sum = tl.zeros((), dtype=tl.float32)
    for i in range(H):
        offs = b * batch_stride + h * qh_stride + s * s_stride + i * d_stride
        x = tl.load(query_ptr + offs)
        x = x.to(tl.float32)
        total_sum += x * x
    tl.store(sum_ptr + pid, total_sum)


# 2) RMSNorm normalization: apply inv_rms = 1/sqrt(mean + eps) and weight
@triton.jit
def rms_norm_kernel(query_ptr, weight_ptr, out_ptr, sum_ptr,
                     B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                     H: tl.constexpr,
                     eps: tl.float32,
                     batch_stride, qh_stride, s_stride, d_stride,
                     out_batch_stride, out_qh_stride, out_s_stride, out_d_stride):
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    h = rem // S
    s = rem % S

    sum_val = tl.load(sum_ptr + pid)
    mean = sum_val / H
    inv_rms = tl.rsqrt(mean + eps)

    for i in range(H):
        in_offs = b * batch_stride + h * qh_stride + s * s_stride + i * d_stride
        out_offs = b * out_batch_stride + h * out_qh_stride + s * out_s_stride + i * out_d_stride
        x = tl.load(query_ptr + in_offs).to(tl.float32)
        w = tl.load(weight_ptr + i).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + out_offs, y)


# 3) Precompute cos/sin for rotation: emb = pos * inv_freq_128, where inv_freq_128 = [inv_freq, inv_freq]
@triton.jit
def precompute_sin_cos_kernel(inv_freq_128_ptr, cos_ptr, sin_ptr,
                               S: tl.constexpr, H: tl.constexpr,
                               pos_ptr,  # int32 [S]
                               num_warps=4):
    t = tl.program_id(0)  # one program per token
    pos = tl.load(pos_ptr + t).to(tl.float32)
    arange = tl.arange(0, H)
    emb = pos * tl.load(inv_freq_128_ptr + arange)
    c = tl.cos(emb)
    s = tl.sin(emb)
    tl.store(cos_ptr + t * H + arange, c)
    tl.store(sin_ptr + t * H + arange, s)


# 4) Apply rotation: y = x * cos + rotate_half(x) * sin, rotate_half: [-x2, x1] using last 64 of x
@triton.jit
def rotate_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                   B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr,
                   H: tl.constexpr,
                   batch_stride_x, h_stride_x, s_stride_x, d_stride_x,
                   cos_s_stride, sin_s_stride,  # cos_ptr/sin_ptr are [S, H], strides accordingly
                   out_batch_stride, out_h_stride, out_s_stride, out_d_stride,
                   num_warps=4):
    pid = tl.program_id(0)
    total = B * num_heads * S
    b = pid // (num_heads * S)
    rem = pid % (num_heads * S)
    h = rem // S
    s = rem % S

    arange = tl.arange(0, H)
    cos_t = tl.load(cos_ptr + s * H + arange).to(tl.float32)  # [H]
    sin_t = tl.load(sin_ptr + s * H + arange).to(tl.float32)  # [H]

    x = tl.load(x_ptr + b * batch_stride_x + h * h_stride_x + s * s_stride_x + arange * d_stride_x).to(tl.float32)  # [H]
    x1 = x[:H//2]  # first 64
    x2 = x[H//2:]  # second 64
    rotated = -x2 * sin_t + x1 * cos_t
    y = x * cos_t + rotated * sin_t
    tl.store(y_ptr + b * out_batch_stride + h * out_h_stride + s * out_s_stride + arange * out_d_stride, y)


# 5) Update caches: write key_rotated and value into cache at positions cache_position[s]
@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        dest_ptr,  # int64 [S]
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                        H: tl.constexpr,
                        batch_stride_kr, h_stride_kr, s_stride_kr, d_stride_kr,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        num_warps=4):
    pid0 = tl.program_id(0)  # over B*S
    pid1 = tl.program_id(1)  # over num_kv_heads
    total = B * S
    b = pid0 // S
    s = pid0 % S
    k = pid1

    dest = tl.load(dest_ptr + s).to(tl.int64)

    arange = tl.arange(0, H)
    key_vals = tl.load(key_rot_ptr + b * batch_stride_kr + k * h_stride_kr + s * s_stride_kr + arange * d_stride_kr).to(tl.float32)
    tl.store(key_cache_ptr + b * batch_stride_kc + k * h_stride_kc + dest * dest_stride + arange * d_stride_kc, key_vals)

    vals = tl.load(value_ptr + b * batch_stride_v + k * h_stride_v + s * s_stride_v + arange * d_stride_v).to(tl.float32)
    tl.store(value_cache_ptr + b * batch_stride_vc + k * h_stride_vc + dest * dest_stride_vc + arange * d_stride_vc, vals)


# -----------------------
# ModelNew.forward
# -----------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value,
                position_ids,  # kept for API; original run doesn't use it in rotation
                key_cache, value_cache,
                cache_position,
                q_norm_weight, k_norm_weight,
                inv_freq,
                rms_norm_eps):
        # Ensure CUDA and contiguity
        assert query.is_cuda and key.is_cuda and value.is_cuda, "All tensors must be on CUDA for Triton."
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()  # int64
        inv_freq = inv_freq.contiguous()  # float32 [64]

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm reduction: sum of squares per (b, h, s)
        sum_query = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_rms_sum = (B * num_q_heads * S,)
        rms_sum_kernel[grid_rms_sum](
            query, sum_query,
            B, num_q_heads, S,
            H,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            num_warps=4
        )

        # 2) RMSNorm normalization: produce query_norm and key_norm (float32 buffers)
        query_norm = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=query.device)
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=key.device)

        grid_rms_norm = (B * num_q_heads * S,)
        rms_norm_kernel[grid_rms_norm](
            query, q_norm_weight, query_norm, sum_query,
            B, num_q_heads, S,
            H,
            float(rms_norm_eps),
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            num_warps=4
        )

        rms_norm_kernel[grid_rms_norm](
            key, k_norm_weight, key_norm, sum_query,
            B, num_kv_heads, S,
            H,
            float(rms_norm_eps),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            num_warps=4
        )

        # 3) Precompute inv_freq_128 and sin/cos for rotation
        inv_freq_128 = torch.empty(H, dtype=torch.float32, device=query.device)
        inv_freq_128[:H//2] = inv_freq
        inv_freq_128[H//2:] = inv_freq  # duplication as in original

        cos = torch.empty((S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((S, H), dtype=torch.float32, device=query.device)

        grid_pre = (S,)
        precompute_sin_cos_kernel[grid_pre](
            inv_freq_128, cos, sin,
            S, H,
            cache_position,
            num_warps=4
        )

        # 4) Rotate query_norm and key_norm
        query_rotated = torch.empty_like(query_norm, dtype=torch.float32, device=query.device)
        key_rotated = torch.empty_like(key_norm, dtype=torch.float32, device=key.device)

        grid_rotate = (B * num_q_heads * S,)
        rotate_kernel[grid_rotate](
            query_norm, cos, sin, query_rotated,
            B, num_q_heads, S,
            H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            H, H,  # cos/sin strides (row-major)
            query_rotated.stride(0), query_rotated.stride(1), query_rotated.stride(2), query_rotated.stride(3),
            num_warps=4
        )

        rotate_kernel[grid_rotate](
            key_norm, cos, sin, key_rotated,
            B, num_kv_heads, S,
            H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            H, H,
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            num_warps=4
        )

        # 5) Update caches: write rotated keys and original values at cache_position[s]
        grid_cache = (B * S, num_kv_heads)
        update_cache_kernel[grid_cache](
            key_rotated, value,
            key_cache, value_cache,
            cache_position,  # int64
            B, num_kv_heads, S,
            H,
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            key_cache.stride(0), key_cache.stride(1), cache_position.stride(0), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), cache_position.stride(0), value_cache.stride(3),
            num_warps=4
        )

        # Return as original function expects
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
