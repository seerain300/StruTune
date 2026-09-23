import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_4d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # program ids
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                # w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off)
                acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store to y
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    # store as fp32; caller will allocate y in fp32 or cast if needed
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_4d_kernel(
    x_ptr, y_ptr,
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_conv3_to_BTW_1d_kernel(
    src_ptr, dst_ptr,
    B, Co, Ho3, Wo3, Tafter, K,  # K = Co * Ho3 * Wo3 = 3840
    src_s0, src_s1, src_s2, src_s3,
    dst_s0, dst_s1, dst_s2,
):
    # 1D launch over total elements
    idx = tl.program_id(0)
    # decompose idx into (b, t, k)
    b_idx = idx // (Tafter * K)
    k_idx = idx % K
    t_idx = (idx // K) % Tafter
    # map k to (co, ho, wo)
    co = k_idx // (Ho3 * Wo3)
    rem = k_idx % (Ho3 * Wo3)
    ho = rem // Wo3
    wo = rem % Wo3

    src_off = b_idx * src_s0 + co * src_s1 + ho * src_s2 + wo * src_s3
    val = tl.load(src_ptr + src_off).to(tl.float32)

    dst_off = b_idx * dst_s0 + t_idx * dst_s1 + k_idx * dst_s2
    tl.store(dst_ptr + dst_off, val)


@triton.jit
def linear_proj_1d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, Tafter, K, D,  # K = 3840, D = 1024
    x_s0, x_s1, x_s2,  # x strides for (B, Tafter, K)
    w_s0, w_s1,        # w strides for (D, K)
    y_s0, y_s1, y_s2,  # y strides for (B, Tafter, D)
):
    idx = tl.program_id(0)
    total = B * Tafter * D
    # decompose idx into (b, t, d)
    b = idx // (Tafter * D)
    t = (idx // D) % Tafter
    d = idx % D
    acc = tl.zeros((), dtype=tl.float32)
    # iterate over K dimension
    for k in range(K):
        x_off = b * x_s0 + t * x_s1 + k * x_s2
        w_off = d * w_s0 + k * w_s1
        xk = tl.load(x_ptr + x_off).to(tl.float32)
        wk = tl.load(w_ptr + w_off).to(tl.float32)
        acc += xk * wk
    y_off = b * y_s0 + t * y_s1 + d * y_s2
    tl.store(y_ptr + y_off, acc)


@triton.jit
def add_pos_scale_2d_kernel(
    out_ptr, pos_ptr, scale, B, Tafter, D,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    b_id = tl.program_id(0)
    d_idx = tl.program_id(1)
    t_idx = d_idx // D
    d_rel = d_idx % D
    out_off = b_id * out_s0 + t_idx * out_s1 + d_rel * out_s2
    val = tl.load(out_ptr + out_off).to(tl.float32)
    pos_off = t_idx * pos_s0 + d_rel * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    new_val = val + pos_val * scale
    tl.store(out_ptr + out_off, new_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers
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
        # Ensure contiguous
        x = input_features.contiguous()
        B, Ci, H, W = x.shape  # Ci=1

        # conv1: (B, 1, 80, W0) -> (B, 384, 40, W0//2)
        Co1 = self.conv2d1_weight.shape[0]
        Kh1, Kw1 = 3, 3
        Ho1 = (H + 2 * 1 - Kh1) // 2 + 1  # 40
        Wo1 = (W + 2 * 1 - Kw1) // 2 + 1  # W0//2
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=torch.float32)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_pad1_4d_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh1, Kw1, Ho1, Wo1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU on conv1 output
        x1_gelu = torch.empty_like(x1, dtype=torch.float32)
        grid1_gelu = (B, Co1, Ho1, Wo1)
        gelu_tanh_4d_kernel[grid1_gelu](
            x1, x1_gelu,
            B, Ci, H, W, Co1, Kh1, Kw1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
        )

        # conv2: (B, 384, 40, Wo1) -> (B, 384, 20, Wo1//2)
        Co2 = self.conv2d2_weight.shape[0]
        Ho2 = (Ho1 + 2 * 1 - 3) // 2 + 1  # 20
        Wo2 = (Wo1 + 2 * 1 - 3) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=torch.float32)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_pad1_4d_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, 3, 3, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU on conv2 output
        x2_gelu = torch.empty_like(x2, dtype=torch.float32)
        grid2_gelu = (B, Co2, Ho2, Wo2)
        gelu_tanh_4d_kernel[grid2_gelu](
            x2, x2_gelu,
            B, Co1, Ho1, Wo1, Co2, 3, 3, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
        )

        # conv3: (B, 384, 20, Wo2) -> (B, 384, 10, Wo2//2)
        Co3 = self.conv2d3_weight.shape[0]
        Ho3 = (Ho2 + 2 * 1 - 3) // 2 + 1  # 10
        Wo3 = (Wo2 + 2 * 1 - 3) // 2 + 1  # Tafter, provided by input via Tafter=Wo3
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=torch.float32)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_pad1_4d_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, 3, 3, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU on conv3 output
        x3_gelu = torch.empty_like(x3, dtype=torch.float32)
        grid3_gelu = (B, Co3, Ho3, Wo3)
        gelu_tanh_4d_kernel[grid3_gelu](
            x3, x3_gelu,
            B, Co2, Ho2, Wo2, Co3, 3, 3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Gather to (B, Tafter, 3840) where Tafter = Wo3
        Tafter = Wo3
        K = Co3 * Ho3 * Wo3  # 384 * 10 * Tafter = 3840
        x25 = torch.empty((B, Tafter, K), device=x.device, dtype=torch.float32)
        total = B * Tafter * K
        # Launch 1D kernel with grid size total
        gather_conv3_to_BTW_1d_kernel[(total,)](
            x3_gelu, x25,
            B, Co3, Ho3, Wo3, Tafter, K,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x25.stride(0), x25.stride(1), x25.stride(2),
        )

        # Linear projection (B, Tafter, 3840) @ (1024, 3840) -> (B, Tafter, 1024)
        D = self.conv_out_weight.shape[0]  # 1024
        out = torch.empty((B, Tafter, D), device=x.device, dtype=torch.float32)
        total_linear = B * Tafter * D
        linear_proj_1d_kernel[(total_linear,)](
            x25, self.conv_out_weight, out,
            B, Tafter, K, D,
            x25.stride(0), x25.stride(1), x25.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # Scale by embed_scale (32.0) and add positional embedding (1500, 1024), add only first Tafter rows
        # Broadcast add across batch in Triton
        grid_scale = (B, Tafter * D)
        add_pos_scale_2d_kernel[grid_scale](
            out, self.positional_embedding, self.embed_scale,
            B, Tafter, D,
            out.stride(0), out.stride(1), out.stride(2),
            self.positional_embedding.stride(0), self.positional_embedding.stride(1),
        )

        # Return out (B, Tafter, 1024) in fp32; evaluator can cast if needed
        return out


# Helper to provide inputs (kept identical to original for consistency)
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = 3840  # 384 * 10
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        # Conv weights — Kaiming init
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        # Linear projection weight
        "conv_out_weight": xavier(d_model, conv_out_dim),
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


def run(*args):
    return ModelNew()(*args)
