import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # One program per output element (b, co, ho, wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and kernel, stride=2, padding=1
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # Store
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr, y_ptr,
    B, Co, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    # GELU tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(c * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_to_long_3d_kernel(
    x_ptr,  # (B, Co, Ho, Wo) float32
    y_ptr,  # (B, Tafter, D) float32, D=Co*Ho*Wo
    B, Co, Ho, Wo, Tafter, D,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2,
):
    # 3D launch over (B, Tafter, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    k_id = tl.program_id(2)

    # Map k_id -> (co, ho, wo) within the last conv output (Co=384, Ho=10, Wo=Tafter)
    # D = Co * Ho * Wo = 384 * 10 * Tafter
    # Note: We compute (co, ho, wo) from k_id; t_id is used only for the batch-time index of the output y.
    # For conv output, k_id enumerates all (co, ho, wo) linearly.
    # We assume Co=384, Ho=10, Wo=Tafter (from the previous conv3 output). We pass D accordingly.
    co = k_id // (Ho * Tafter)
    rem = k_id % (Ho * Tafter)
    ho = rem // Tafter
    wo = rem % Tafter

    x_off = b_id * x_s0 + co * x_s1 + ho * x_s2 + wo * x_s3
    val = tl.load(x_ptr + x_off).to(tl.float32)

    y_off = b_id * y_s0 + t_id * y_s1 + k_id * y_s2
    tl.store(y_ptr + y_off, val)


@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, y_ptr,
    B, T, D, K,
    x_s0, x_s1, x_s2,  # (B, T, K) float32
    w_s0, w_s1,        # (D, K) float32
    y_s0, y_s1, y_s2,  # (B, T, D) float32
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, acc)


@triton.jit
def add_pos_embedding_kernel(
    y_ptr, pos_ptr, scale, B, T, D,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    y_val += pos_val * scale

    tl.store(y_ptr + y_off, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()


def run(*args):
    return ModelNew()(*args)
