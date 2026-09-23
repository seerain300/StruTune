import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,        # *bf16 or *f16, (B, Ci, H, W)
    w_ptr,        # *bf16 or *f16, (Co, Ci, Kh, Kw)
    b_ptr,        # *bf16 or *f16, (Co,)
    y_ptr,        # *bf16 or *f16, (B, Co, Ho, Wo)
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
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

    # Sum over input channels and kernel window
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
    tl.store(y_ptr + y_off, acc)  # store as fp32; y_ptr dtype will cast as needed


@triton.jit
def gelu_tanh_kernel(
    x_ptr,  # *bf16 or *f16, (B, Co, Ho, Wo) after conv
    y_ptr,  # *bf16 or *f16, (B, Co, Ho, Wo)
    B, Co, Ho, Wo,
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

    # tanh-based GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_x25_kernel(
    x3_gelu_ptr,    # *bf16 or *f16, (B, Co, Ho, Wo) where Co=384, Ho=10, Wo=Tafter
    xgather_ptr,    # *bf16 or *f16, (B, Tafter, K=3840)
    B, Co, Ho, Wo, Tafter, K,
    xg_s0, xg_s1, xg_s2,  # strides for (B, Tafter, K)
    x3_s0, x3_s1, x3_s2, x3_s3,
):
    # Grid: (B, Tafter, K)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    k_id = tl.program_id(2)

    # Map k -> (co, ho)
    co = k_id // Ho
    ho = k_id % Ho

    x_off = b_id * x3_s0 + co * x3_s1 + ho * x3_s2 + t_id * x3_s3
    val = tl.load(x3_gelu_ptr + x_off)
    y_off = b_id * xg_s0 + t_id * xg_s1 + k_id * xg_s2
    tl.store(xgather_ptr + y_off, val)


@triton.jit
def linear_proj_kernel(
    xg_ptr,        # *bf16 or *f16, (B, Tafter, K=3840)
    wout_ptr,      # *bf16 or *f16, (D=1024, K=3840)
    out_ptr,       # *bf16 or *f16, (B, Tafter, D=1024)
    B, T, K, D,
    xg_s0, xg_s1, xg_s2,
    wout_s0, wout_s1,
    out_s0, out_s1, out_s2,
):
    # Grid: (B, Tafter, 1024)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks (64) for performance
    for k_start in range(0, K, 64):
        for kk in range(64):
            k = k_start + kk
            mask_k = k < K
            x_off = b_id * xg_s0 + t_id * xg_s1 + k * xg_s2
            x_val = tl.load(xg_ptr + x_off, mask=mask_k, other=0.0).to(tl.float32)
            w_off = d_id * wout_s0 + k * wout_s1
            w_val = tl.load(wout_ptr + w_off).to(tl.float32)
            acc += x_val * w_val
    # Store
    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def scale_kernel(
    out_ptr,      # *bf16 or *f16, (B, Tafter, D)
    scale,        # float32 scalar
    B, T, D,
    out_s0, out_s1, out_s2,
):
    # Grid: (B, Tafter, 1024)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    val = tl.load(out_ptr + out_off).to(tl.float32)
    val = val * scale
    tl.store(out_ptr + out_off, val)


@triton.jit
def add_pos_embedding_kernel(
    out_ptr,       # *bf16 or *f16, (B, Tafter, D=1024)
    pos_ptr,       # *bf16 or *f16, (P=1500, D=1024) but we only use first T rows
    B, T, D, P,    # P is max positions; pos has shape (P, D)
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    # Grid: (B, Tafter, 1024)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # out_off is scalar for this (b, t, d)
    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)
    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    out_val = out_val + pos_val
    tl.store(out_ptr + out_off, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers so they move with .to(device)
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
        x0 = input_features.contiguous()
        B, Ci, H, W = x0.shape  # Ci=1

        # Stage 1: Conv1 (1 -> 384 channels), stride=2, padding=1
        Co1 = self.conv2d1_weight.shape[0]
        Ho1 = (H + 2 * 1 - 3) // 2 + 1
        Wo1 = (W + 2 * 1 - 3) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x0.device, dtype=x0.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x0, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, 3, 3, Ho1, Wo1,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU on conv1
        x1_gelu = torch.empty_like(x1, dtype=x1.dtype)
        gelu_tanh_kernel[(B, Co1, Ho1, Wo1)](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
        )

        # Stage 2: Conv2 (384 -> 384), stride=2, padding=1
        Co2 = self.conv2d2_weight.shape[0]
        Ho2 = (Ho1 + 2 * 1 - 3) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - 3) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x1_gelu.device, dtype=x1_gelu.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, 3, 3, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU on conv2
        x2_gelu = torch.empty_like(x2, dtype=x2.dtype)
        gelu_tanh_kernel[(B, Co2, Ho2, Wo2)](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
        )

        # Stage 3: Conv3 (384 -> 384), stride=2, padding=1
        Co3 = self.conv2d3_weight.shape[0]
        Ho3 = (Ho2 + 2 * 1 - 3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - 3) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x2_gelu.device, dtype=x2_gelu.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, 3, 3, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU on conv3
        x3_gelu = torch.empty_like(x3, dtype=x3.dtype)
        gelu_tanh_kernel[(B, Co3, Ho3, Wo3)](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Gather to form (B, Tafter, 3840): k = co*10 + ho
        Tafter = Wo3  # time dimension after last conv
        K = Co3 * Ho3 * Tafter  # 384 * 10 * Tafter = 3840 if the math in the original code holds
        xgather = torch.empty((B, Tafter, K), device=x3_gelu.device, dtype=x3_gelu.dtype)
        grid_gather = (B, Tafter, K)
        gather_x25_kernel[grid_gather](
            x3_gelu, xgather,
            B, Co3, Ho3, Tafter, K,
            xgather.stride(0), xgather.stride(1), xgather.stride(2),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Linear projection to d_model=1024
        D = self.conv_out_weight.shape[0]  # 1024
        out_linear = torch.empty((B, Tafter, D), device=xgather.device, dtype=xgather.dtype)
        grid_linear = (B, Tafter, D)
        linear_proj_kernel[grid_linear](
            xgather, self.conv_out_weight, out_linear,
            B, Tafter, K, D,
            xgather.stride(0), xgather.stride(1), xgather.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            out_linear.stride(0), out_linear.stride(1), out_linear.stride(2),
        )

        # Scale by embed_scale
        grid_scale = (B, Tafter, D)
        scale_kernel[grid_scale](
            out_linear, self.embed_scale,
            B, Tafter, D,
            out_linear.stride(0), out_linear.stride(1), out_linear.stride(2),
        )

        # Add positional embedding: broadcast add per (t, d), shared across batch
        P = self.positional_embedding.shape[0]  # 1500
        grid_add = (B, Tafter, D)
        add_pos_embedding_kernel[grid_add](
            out_linear, self.positional_embedding,
            B, Tafter, D, P,
            out_linear.stride(0), out_linear.stride(1), out_linear.stride(2),
            self.positional_embedding.stride(0), self.positional_embedding.stride(1),
        )

        return out_linear


def run(*args):
    return ModelNew()(*args)
