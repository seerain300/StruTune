import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # Grid: (B, Co, Ho, Wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, padding=1
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

    # Store to y[b, co, ho, wo]
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_erf_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # Grid: (B, Co, Ho, Wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gelu = 0.5 * x_val * (1.0 + tl.libdevice.erf(x_val * inv_sqrt2))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def linear_proj_kernel(
    x25_ptr, w_ptr, out_ptr,
    B: tl.constexpr, Tafter: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    x25_s0, x25_s1,  # x25 is laid out as [B*Tafter, K]
    w_s0, w_s1,      # w is laid out as [D, K]
    out_s0, out_s1, out_s2,  # out is laid out as [B*Tafter, D]
):
    # Grid: (B, Tafter, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    base_out = b_id * Tafter + t_id
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        x_off = base_out * x25_s0 + k * x25_s1
        x_val = tl.load(x25_ptr + x_off).to(tl.float32)
        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    out_off = base_out * out_s0 + d_id * out_s1
    tl.store(out_ptr + out_off, acc)


@triton.jit
def add_pos_kernel(
    y_ptr, pos_ptr,
    B: tl.constexpr, Tafter: tl.constexpr, D: tl.constexpr,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,
):
    # Grid: (B, Tafter, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    tl.store(y_ptr + y_off, y_val + pos_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                 conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024,


def run(*args):
    return ModelNew()(*args)
