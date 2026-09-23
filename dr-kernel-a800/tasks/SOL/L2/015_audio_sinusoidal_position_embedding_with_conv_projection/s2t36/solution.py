import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W_in, OC, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
):
    # program ids
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # iterate over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = oh * stride_h + kh - pad_h
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = ow * stride_w + kw - pad_w
                valid_w = (iw >= 0) & (iw < W_in)
                in_bounds = valid_h & valid_w
                # compute input linear index: ((b * C_in + ic) * H + ih) * W_in + iw
                x_idx = (((b * C_in + ic) * H + ih) * W_in + iw)
                x_val = tl.load(X_ptr + x_idx, mask=in_bounds, other=0.0)
                # weight index: W_ptr layout [OC, C_in, 3, 3] -> w[oc, ic, kh, kw]
                w_idx = oc * (C_in * 9) + ic * 9 + kh * 3 + kw
                w_val = tl.load(W_ptr + w_idx)
                acc += x_val * w_val

    # add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc = acc + bias_val

    # store output: Y layout [B, OC, H_out, W_out] -> ((b * OC + oc) * H_out + oh) * W_out + ow
    y_idx = (((b * OC + oc) * H_out + oh) * W_out + ow)
    tl.store(Y_ptr + y_idx, acc)


@triton.jit
def gelu_kernel_1d(X_ptr, Y_ptr, N, scale: tl.float32, approximate: tl.int32):
    # approximate=1 for tanh GELU (fast), 0 would be erf-based. Here we use tanh.
    idx = tl.program_id(0)
    x = tl.load(X_ptr + idx)
    # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    y = gelu * scale
    tl.store(Y_ptr + idx, y)


@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M, scale: tl.float32,
):
    # grid over (B, T, tiles of M)
    b = tl.program_id(0)
    t = tl.program_id(1)
    m0 = tl.program_id(2)
    m_offsets = m0 * 128 + tl.arange(0, 128)
    mask_m = m_offsets < M

    acc = tl.zeros([128], dtype=tl.float32)
    # loop over N in chunks
    for n0 in range(0, N, 256):
        n_offsets = n0 + tl.arange(0, 256)
        mask_n = n_offsets < N
        # X[b, t, n_offsets] -> linear index: b*(T*N) + t*N + n_offsets
        x_ptrs = X_ptr + b * (T * N) + t * N + n_offsets
        x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [256]
        # W[m_offsets, n_offsets] -> [128, 256]
        w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    acc = acc * scale
    pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
    acc = acc + pos_vec

    y_ptrs = Y_ptr + b * (T * M) + t * M + m_offsets
    tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), dtype=bfloat16
        conv2d1_weight: (384, 1, 3, 3), dtype=bfloat16
        conv2d1_bias: (384), dtype=bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3), dtype=bfloat16
        conv2d2_bias, conv2d3_bias: (384), dtype=bfloat16
        conv_out_weight: (1024, 3840), dtype=bfloat16 (d_model=1024, in_features=384*10)
        positional_embedding: (1500, 1024), dtype=bfloat16
        embed_scale: float, e.g., sqrt(1024)=32.0
        """
        B, C, H, W_in = input_features.shape
        stride = 2
        pad = 1

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = 384
        H_out1 = (H - 1) // stride + 1
        W_out1 = (W_in - 1) // stride + 1
        x1 = torch.empty((B, OC1, H_out1, W_out1), device=input_features.device, dtype=torch.float32)
        grid1 = (B, OC1, H_out1, W_out1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C, H, W_in, OC1, H_out1, W_out1,
            stride, stride, pad, pad,
        )
        # GELU
        x1_flat = x1.flatten()
        x1_gelu = torch.empty_like(x1_flat, dtype=torch.float32)
        N1 = x1_flat.numel()
        gelu_kernel_1d[(1,)](x1_flat, x1_gelu, N1, 1.0, 1)
        x1 = x1_gelu.view(B, OC1, H_out1, W_out1)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        OC2 = 384
        H_out2 = (H_out1 - 1) // stride + 1
        W_out2 = (W_out1 - 1) // stride + 1
        x2 = torch.empty((B, OC2, H_out2, W_out2), device=input_features.device, dtype=torch.float32)
        grid2 = (B, OC2, H_out2, W_out2)
        conv2d_stride2_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, OC1, H_out1, W_out1, OC2, H_out2, W_out2,
            stride, stride, pad, pad,
        )
        x2_flat = x2.flatten()
        x2_gelu = torch.empty_like(x2_flat, dtype=torch.float32)
        N2 = x2_flat.numel()
        gelu_kernel_1d[(1,)](x2_flat, x2_gelu, N2, 1.0, 1)
        x2 = x2_gelu.view(B, OC2, H_out2, W_out2)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        OC3 = 384
        H_out3 = (H_out2 - 1) // stride + 1
        W_out3 = (W_out2 - 1) // stride + 1
        x3 = torch.empty((B, OC3, H_out3, W_out3), device=input_features.device, dtype=torch.float32)
        grid3 = (B, OC3, H_out3, W_out3)
        conv2d_stride2_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, OC2, H_out2, W_out2, OC3, H_out3, W_out3,
            stride, stride, pad, pad,
        )
        x3_flat = x3.flatten()
        x3_gelu = torch.empty_like(x3_flat, dtype=torch.float32)
        N3 = x3_flat.numel()
        gelu_kernel_1d[(1,)](x3_flat, x3_gelu, N3, 1.0, 1)
        x3 = x3_gelu.view(B, OC3, H_out3, W_out3)

        # Reshape: (B, channels, freq, time) -> (B, time, channels*freq)
        B, C3, F_out, T_out = x3.shape  # C3=384, F_out=40, T_out = time_after_conv
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(B, T_out, C3 * F_out)

        # Linear projection to d_model=1024 (no bias) + positional embedding
        X = x3  # (B, T_out, 3840)
        B2, T2, N = X.shape
        M = conv_out_weight.shape[0]  # 1024
        Y = torch.empty((B2, T2, M), device=input_features.device, dtype=torch.float32)
        grid_linear = (B2, T2, (M + 128 - 1) // 128)
        linear_project_pos_kernel[grid_linear](
            X, conv_out_weight, positional_embedding, Y,
            B2, T2, N, M, float(embed_scale),
        )

        return Y


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = downsample_hidden_size * 10  # 3840
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(in_f, out_f):
        # Return (in_f, out_f) so weight can be transposed to (out_f, in_f)
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
        # Linear projection weight: (out=1024, in=384*10=3840)
        "conv_out_weight": xavier(conv_out_dim, d_model).t().to(dtype),  # (1024, 3840)
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


def run(*args):
    return ModelNew()(*args)
