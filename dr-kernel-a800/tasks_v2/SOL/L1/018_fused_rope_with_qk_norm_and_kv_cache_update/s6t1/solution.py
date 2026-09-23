import torch
import triton
import triton.language as tl


# -----------------------
# Triton kernels
# -----------------------

# 1) RMSNorm: compute sum of squares across head_dim for each (b, h, s)
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


# 2) RMSNorm: normalize and apply weight (per-dim)
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


# 3) Precompute cos/sin for rotation: emb = pos * inv_freq_128, cos/ sin over 128 dims
#    inv_freq_128 is [128] on host; we pass cos_ptr, sin_ptr as 2D [S, 128] buffers.
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


# 4) Apply rotation: y = x * cos + rotate_half(x) * sin, rotate_half: [-x2, x1]
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
                position_ids,  # kept for API; original does not use it in rotation
                key_cache, value_cache,
                cache_position,
                q_norm_weight, k_norm_weight,
                inv_freq,
                rms_norm_eps):
        # Shapes and contiguity
        B, num_q_heads, S, H = query.shape
        _, num


def run(*args):
    return ModelNew()(*args)
