import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_bias_kernel(
    x_ptr,          # *f16 or *bf16, (B, Ci, H, W)
    w_ptr,          # *f16 or *bf16, (Co, Ci, Kh, Kw)
    b_ptr,          # *f16 or *bf16, (Co,)
    y_ptr,          # *f16 or *bf16, (B, Co, Ho, Wo)
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # grid: (B, Co, Ho, Wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over input channels and kernel
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

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store output (cast to original dtype of y_ptr)
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,  # *f16 or *bf16
    y_ptr,  # *f16 or *bf16
    B, Co, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # grid: (B, Co, Ho, Wo)
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
def gather_conv3_to_k_kernel(
    x3_gelu_ptr,    # *f16 or *bf16, (B, Co, Ho, Wo), Co=384, Ho=10, Wo=Tafter
    xg_ptr,         # *f16 or *bf16, (B, T, K), K=Co*Ho*Wo
    B, Co, Ho, Wo, Tafter,
    xg_s0, xg_s1, xg_s2,  # strides for (B, T, K)
    xg_s3, xg_s4, xg_s5,  # additional strides if needed (unused)
    x3_s0, x3_s1, x3_s2, x3_s3,
):
    # grid: (B, Tafter, 3840)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    k_id = tl.program_id(2)

    # compute co and ho from k
    co = k_id // (Ho * Wo)
    rem = k_id % (Ho * Wo)
    ho = rem // Wo
    wo = rem % Wo

    x_off = b_id * x3_s0 + co * x3_s1 + ho * x3_s2 + wo * x3_s3
    x_val = tl.load(x3_gelu_ptr + x_off).to(tl.float32)

    # write to xg[b, t, k]
    xg_off = b_id * xg_s0 + t_id * xg_s1 + k_id * xg_s2
    tl.store(xg_ptr + xg_off, x_val)


@triton.jit
def linear_proj_kernel(
    xg_ptr,     # *f16 or *bf16, (B, T, K=3840)
    w_ptr,      # *f16 or *bf16, (D=1024, K=3840)
    out_ptr,    # *f16 or *bf16, (B, T, D)
    B, T, K, D,
    xg_s0, xg_s1, xg_s2,
    w_s0, w_s1,
    out_s0, out_s1, out_s2,
):
    # grid: (B, T, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # sum over K
    for k in range(0, K):
        x_off = b_id * xg_s0 + t_id * xg_s1 + k * xg_s2
        x_val = tl.load(xg_ptr + x_off).to(tl.float32)

        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)

        acc += x_val * w_val

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def scale_add_pos_kernel(
    out_ptr,        # *f16 or *bf16, (B, T, D)
    scale,          # float32
    pos_ptr,        # *f16 or *bf16, (P=1500, D=1024)
    B, T, D, P,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    # grid: (B, T, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    new_val = out_val * scale + pos_val
    tl.store(out_ptr + out_off, new_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers to mimic original module structure
        self.register_buffer("conv2d1_weight", conv2d1_weight)   # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)       # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)   # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)       # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)   # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)       # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight) # (1024, 3840)
        self.register_buffer("positional_embedding", positional_embedding) # (1500, 1024), bfloat16
        self.embed_scale = float(embed_scale)  # sqrt(1024) = 32.0

    def forward(self, input_features):
        # Ensure input is contiguous (bf16 as per get_inputs)
        x = input_features.contiguous()
        B, Ci, H, W = x.shape  # Ci=1

        # conv1: (1 -> 384), stride=2, padding=1
        Co1, Ci1, Kh1, Kw1 = self.conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh1) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw1) // 2 + 1

        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=x.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_bias_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh1, Kw1, Ho1, Wo1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4, num_stages=2
        )

        # gelu conv1
        x1_gelu = torch.empty_like(x1)
        gelu_tanh_kernel[(B, Co1, Ho1, Wo1)](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            num_warps=4, num_stages=2
        )

        # conv2: (384 -> 384), stride=2, padding=1
        Co2, Ci2, Kh2, Kw2 = self.conv2d2_weight.shape
        Ho2 = (Ho1 + 2 * 1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1

        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=x.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_bias_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4, num_stages=2
        )

        # gelu conv2
        x2_gelu = torch.empty_like(x2)
        gelu_tanh_kernel[(B, Co2, Ho2, Wo2)](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            num_warps=4, num_stages=2
        )

        # conv3: (384 -> 384), stride=2, padding=1
        Co3, Ci3, Kh3, Kw3 = self.conv2d3_weight.shape
        Ho3 = (Ho2 + 2 * 1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1

        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=x.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_bias_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4, num_stages=2
        )

        # gelu conv3
        x3_gelu = torch.empty_like(x3)
        gelu_tanh_kernel[(B, Co3, Ho3, Wo3)](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            num_warps=4, num_stages=2
        )

        # Determine Tafter
        Tafter = Wo3  # conv3 produces (B, 384, 10, Tafter), where Wo3 is final time dimension

        # Prepare x_gather: (B, Tafter, K=Co3*Ho3*Wo3) where Co3=384, Ho3=10, Wo3=Tafter -> K=3840
        K = Co3 * Ho3 * Wo3
        xg = torch.empty((B, Tafter, K), device=x.device, dtype=x.dtype)
        # Launch gather kernel: grid over (B, Tafter, K)
        gather_conv3_to_k_kernel[(B, Tafter, K)](
            x3_gelu, xg,
            B, Co3, Ho3, Wo3, Tafter,
            xg.stride(0), xg.stride(1), xg.stride(2),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2),
            num_warps=4, num_stages=2
        )

        # Linear projection: (B, Tafter, 1024) = xg @ conv_out_weight^T
        D = self.conv_out_weight.shape[0]  # 1024
        K_proj = self.conv_out_weight.shape[1]  # 3840
        out = torch.empty((B, Tafter, D), device=x.device, dtype=x.dtype)
        linear_proj_kernel[(B, Tafter, D)](
            xg, self.conv_out_weight, out,
            B, Tafter, K_proj, D,
            xg.stride(0), xg.stride(1), xg.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        scale = self.embed_scale  # 32.0
        scaled = torch.empty_like(out)
        scale_add_pos_kernel[(B, Tafter, D)](
            out, scale, self.positional_embedding,  # positional_embedding is (P=1500, D=1024)
            B, Tafter, D, 1500,  # P is 1500, but we only use first Tafter rows
            scaled.stride(0), scaled.stride(1), scaled.stride(2),
            self.positional_embedding.stride(0), self.positional_embedding.stride(1),
            num_warps=4, num_stages=2
        )

        # Broadcast add positional embedding: out[b, t, d] += pos[t, d]
        # We already added in scaled kernel; out is now scaled + pos

        # Return the result
        return scaled


def run(*args):
    return ModelNew()(*args)
