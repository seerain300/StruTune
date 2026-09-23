import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,         # *f16 or *bf16: input (B, Ci, H, W)
    w_ptr,         # *f16 or *bf16: weight (Co, Ci, Kh, Kw)
    b_ptr,         # *f32: bias (Co,)
    y_ptr,         # *f16 or *bf16: output (B, Co, Ho, Wo)
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,       # strides for x
    w_s0, w_s1, w_s2, w_s3,       # strides for w
    y_s0, y_s1, y_s2, y_s3,       # strides for y
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, padding=1
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
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_erf_kernel(
    x_ptr,  # *f16 or *bf16
    y_ptr,  # *f16 or *bf16
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
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu = 0.5 * x_val * (1.0 + tl.libdevice.erf(x_val * 0.7071067811865476))
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_to_BTD_kernel(
    x_ptr,     # *f16 or *bf16: input conv3_gelu (B, 384, 10, Tafter)
    y_ptr,     # *f16 or *bf16: output (B, Tafter, 3840)
    B, Tafter,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2,
):
    # Grid: (B, 1, 3840)
    b_id = tl.program_id(0)
    d_id = tl.program_id(2)  # d in [0, 3840)
    # Decode d into (co, t_idx)
    co = d_id // 10
    t_idx = d_id % 10
    # Loop over Tafter to fill rows
    for t in range(Tafter):
        x_off = b_id * x_s0 + co * x_s1 + 10 * x_s2 + t_idx * x_s3
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        y_off = b_id * y_s0 + t * y_s1 + d_id * y_s2
        tl.store(y_ptr + y_off, x_val)


@triton.jit
def linear_proj_kernel(
    x_ptr,       # *f16 or *bf16: (B, Tafter, 3840)
    w_ptr,       # *f16 or *bf16: (1024, 3840)
    y_ptr,       # *f16 or *bf16: (B, Tafter, 1024)
    B, Tafter, K,
    x_s0, x_s1, x_s2,
    w_s0, w_s1,
    y_s0, y_s1, y_s2,
):
    # Grid: (B, Tafter, 1024)
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
def scale_kernel(
    x_ptr, y_ptr, B, Tafter, D, scale,
    x_s0, x_s1, x_s2,
    y_s0, y_s1, y_s2,
):
    # Grid: (B, Tafter, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)
    x_off = b_id * x_s0 + t_id * x_s1 + d_id * x_s2
    x_val = tl.load(x_ptr + x_off).to(tl.float32) * scale
    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, x_val)


@triton.jit
def add_pos_embedding_kernel(
    x_ptr, pos_ptr, B, Tafter, D,
    x_s0, x_s1, x_s2,
    pos_s0, pos_s1,
):
    # Grid: (B, Tafter, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)
    # Load pos[t, d] with 64-bit indexing
    pos_off = (t_id.to(tl.int64)) * 1024 + (d_id.to(tl.int64))
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    x_off = b_id * x_s0 + t_id * x_s1 + d_id * x_s2
    x_val = tl.load(x_ptr + x_off).to(tl.float32) + pos_val
    tl.store(x_ptr + x_off, x_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register weights as buffers to ensure they move with .to(device)
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840)
        self.register_buffer("positional_embedding", positional_embedding)  # (1500, 1024)
        self.embed_scale = float(embed_scale)  # sqrt(1024) = 32.0

    def forward(self, input_features):
        # Ensure input is contiguous
        x0 = input_features.contiguous()  # (B, 1, 80, T0)
        B, Ci, H, W = x0.shape

        # conv1: output (B, 384, 40, W1) with W1 = W//2
        Co1 = self.conv2d1_weight.shape[0]
        Kh1, Kw1 = 3, 3
        Ho1 = (H + 2 * 1 - Kh1) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw1) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x0.device, dtype=x0.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x0, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh1, Kw1, Ho1, Wo1,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=1, num_stages=1,
        )

        # conv2
        Co2 = self.conv2d2_weight.shape[0]
        Kh2, Kw2 = 3, 3
        Ho2 = (Ho1 + 2 * 1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x0.device, dtype=x0.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=1, num_stages=1,
        )

        # conv3
        Co3 = self.conv2d3_weight.shape[0]  # 384
        Kh3, Kw3 = 3, 3
        Ho3 = (Ho2 + 2 * 1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x0.device, dtype=x0.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=1, num_stages=1,
        )

        # GELU on conv3 output
        x3_gelu = torch.empty_like(x3)  # same dtype as x3 (bfloat16)
        grid_gelu = (B, Co3, Ho3, Wo3)
        gelu_erf_kernel[grid_gelu](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            num_warps=1, num_stages=1,
        )

        # Gather to (B, Tafter, 3840): Tafter = Wo3
        Tafter = int(Wo3)
        x_gather = torch.empty((B, Tafter, 384 * 10), device=x0.device, dtype=x0.dtype)
        grid_gather = (B, 1, 384 * 10)
        gather_to_BTD_kernel[grid_gather](
            x3_gelu, x_gather,
            B, Tafter,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x_gather.stride(0), x_gather.stride(1), x_gather.stride(2),
            num_warps=1, num_stages=1,
        )

        # Linear projection: (B, Tafter, 3840) @ (1024, 3840) -> (B, Tafter, 1024)
        y_linear = torch.empty((B, Tafter, 1024), device=x0.device, dtype=x0.dtype)
        grid_linear = (B, Tafter, 1024)
        linear_proj_kernel[grid_linear](
            x_gather, self.conv_out_weight, y_linear,
            B, Tafter, 3840,
            x_gather.stride(0), x_gather.stride(1), x_gather.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            y_linear.stride(0), y_linear.stride(1), y_linear.stride(2),
            num_warps=1, num_stages=1,
        )

        # Scale by embed_scale
        y_scaled = torch.empty_like(y_linear)
        grid_scale = (B, Tafter, 1024)
        scale_kernel[grid_scale](
            y_linear, y_scaled,
            B, Tafter, 1024, self.embed_scale,
            y_linear.stride(0), y_linear.stride(1), y_linear.stride(2),
            y_scaled.stride(0), y_scaled.stride(1), y_scaled.stride(2),
            num_warps=1, num_stages=1,
        )

        # Add positional embedding: pos is (1500, 1024). Broadcast add across batch.
        # Only the first Tafter rows are used.
        grid_add = (B, Tafter, 1024)
        add_pos_embedding_kernel[grid_add](
            y_scaled, self.positional_embedding,
            B, Tafter, 1024,
            y_scaled.stride(0), y_scaled.stride(1), y_scaled.stride(2),
            self.positional_embedding.stride(0), self.positional_embedding.stride(1),
            num_warps=1, num_stages=1,
        )

        return y_scaled


def run(*args):
    return ModelNew()(*args)
