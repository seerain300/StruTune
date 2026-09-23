import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_bias_kernel(
    x_ptr,  # *bf16
    w_ptr,  # *bf16
    b_ptr,  # *bf16
    y_ptr,  # *bf16
    B: tl.constexpr,
    Ci: tl.constexpr,  # input channels
    H, W,                # input spatial dims (int32)
    Co,                  # output channels
    Kh: tl.constexpr, Kw: tl.constexpr,  # kernel size (3x3)
    Ho, Wo,              # output spatial dims (int32)
    x_s0, x_s1, x_s2, x_s3,  # strides for x: N, C_in, H, W
    w_s0, w_s1, w_s2, w_s3,  # strides for w: C_out, C_in, Kh, Kw
    y_s0, y_s1, y_s2, y_s3,  # strides for y: N, C_out, Ho, Wo
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off)
                acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)  # acc is fp32; y_ptr is bf16 (implicit cast on store)


@triton.jit
def gelu_erf_kernel(
    x_ptr,  # *bf16
    y_ptr,  # *bf16
    B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    inv_sqrt2 = 0.7071067811865475  # 1/sqrt(2)
    gelu = 0.5 * x_val * (1.0 + tl.math.erf(x_val * inv_sqrt2))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def linear_proj_kernel(
    x_ptr,  # *bf16, shape (B, Tafter, K)
    w_ptr,  # *bf16, shape (D, K) where D=1024, K=3840
    y_ptr,  # *bf16, shape (B, Tafter, D)
    B, Tafter, K, D,
    x_s0, x_s1, x_s2,  # strides for x: N, T, K
    w_s0, w_s1,         # strides for w: D, K
    y_s0, y_s1, y_s2,   # strides for y: N, T, D
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for k in range(0, K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        xk = tl.load(x_ptr + x_off).to(tl.float32)
        wk_off = d_id * w_s0 + k * w_s1
        wk = tl.load(w_ptr + wk_off).to(tl.float32)
        acc += xk * wk

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, acc)  # acc is fp32; y_ptr is bf16 (implicit cast)


@triton.jit
def scale_kernel(
    x_ptr,  # *bf16
    y_ptr,  # *bf16
    B, T, D,
    x_s0, x_s1, x_s2,
    y_s0, y_s1, y_s2,
    scale: tl.constexpr,  # scalar, e.g., 32.0
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    x_off = b_id * x_s0 + t_id * x_s1 + d_id * x_s2
    x_val = tl.load(x_ptr + x_off).to(tl.float32)
    y_val = x_val * scale
    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, y_val)


@triton.jit
def add_pos_embedding_kernel(
    y_ptr,    # *bf16, shape (B, Tafter, D)
    pos_ptr,  # *f32, shape (max_pos=1500, D=1024)
    B, Tafter, D,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,  # pos strides in int64
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off)  # pos_ptr is fp32

    y_val += pos_val
    tl.store(y_ptr + y_off, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                 conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers; we won't use torch ops in forward.
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3), bf16
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,), bf16
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3), bf16
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,), bf16
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3), bf16
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,), bf16
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840), bf16
        self.register_buffer("positional_embedding", positional_embedding)  # (1500, 1024), fp32
        self.embed_scale = float(embed_scale)  # 32.0

    def forward(self, input_features):
        # input_features: (B, 1, 80, time_dim), bf16, contiguous
        x = input_features
        B, Ci, H, W = x.shape
        assert Ci == 1, "This implementation expects input channels=1"

        # conv1: (1, 80, W) -> (B, 384, 40, W//2)
        Co1, Ci1, Kh, Kw = self.conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1  # padding=1, stride=2
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=x.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_bias_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh, Kw, Ho1, Wo1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4, num_stages=2,
        )

        # conv2: (384, 40, W1) -> (B, 384, 20, W1//2)
        Co2, Ci2, Kh, Kw = self.conv2d2_weight.shape
        assert Co2 == 384 and Ci2 == 384
        H2 = Ho1
        W2 = Wo1 // 2
        x2 = torch.empty((B, Co2, H2, W2), device=x.device, dtype=x.dtype)
        grid2 = (B, Co2, H2, W2)
        conv2d_stride2_bias_kernel[grid2](
            x1, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Ci2, H2, W2, Co2, Kh, Kw, H2, W2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4, num_stages=2,
        )

        # conv3: (384, 20, W2) -> (B, 384, 10, W2//2)
        Co3, Ci3, Kh, Kw = self.conv2d3_weight.shape
        assert Co3 == 384 and Ci3 == 38


def run(*args):
    return ModelNew()(*args)
