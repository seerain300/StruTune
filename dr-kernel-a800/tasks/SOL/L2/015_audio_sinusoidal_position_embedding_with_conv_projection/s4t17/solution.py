import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# X: (B, C_in, H, W), W: (C_out, C_in, 3, 3), BIAS: (C_out)
# Y: (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)

    # Decode flattened HW output index
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    # Tile over output channels
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Reduction over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = h_out * 2 + kh - 1  # padding=1
                in_w = w_out * 2 + kw - 1
                # Valid if within input bounds
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)

                # Load X[b, ci, in_h, in_w] vector over CO tile
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                # Load W[co_offsets, ci, kh, kw] vector over CO tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # Accumulate
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y
    y_ptrs = Y + pid_b * stride_yb + co_offsets * stride_yc + h_out * stride_yh + w_out * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# Triton kernel: GELU (tanh approximation), elementwise over N elements
@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, gelu, mask=mask)


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K), W: (M, K), Y: (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,   # tile over output channels M
    BLOCK_K: tl.constexpr    # tile over reduction K
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets] -> vector (BLOCK_K,)
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[:, k_offsets] for M tile -> matrix (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k w_mat[m, k] * x_vec[k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store results Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add scaled positional embedding to Y
# Y: (B, T_after, M), POS: (1500, M), scale: float
@triton.jit
def add_scaled_pos_emb_kernel(
    Y, POS, scale,
    B, T_after, M,
    stride_yb, stride_yt, stride_ym,
    stride_pos0, stride_pos1,  # POS is 2D (T_after, M)
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * 64
    m_offsets = m_start + tl.arange(0, 64)
    m_mask = m_offsets < M

    # Load Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    y_vals = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load POS[t, m_offsets]
    pos_ptrs = POS + t * stride_pos0 + m_offsets * stride_pos1
    pos_vals = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    y_vals += scale * pos_vals

    # Store back
    tl.store(y_ptrs, y_vals, mask=m_mask)


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

    # Sinusoidal positional embedding (PyTorch-generated; Triton will add scaled slice)
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    positional_embedding = pe.to(dtype)

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
        "positional_embedding": positional_embedding,
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    B, C_in, H1, W1 = input_features.shape
    device = input_features.device
    dtype = input_features.dtype

    # Stage 1: Conv2d (1 -> 384) + GELU
    C_out1 = conv2d1_weight.shape[0]
    H_out1 = (H1 - 3) // 2 + 1
    W_out1 = (W1 - 3) // 2 + 1
    y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)
    x1 = input_features
    grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
    conv2d_3x3_stride2_pad1_kernel[grid1](
        x1, conv2d1_weight, conv2d1_bias, y1,
        B, 1, H1, W1, C_out1, H_out1, W_out1,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        BLOCK_CO=64,
    )
    y1_gelu = torch.empty_like(y1)
    N1 = y1.numel()
    gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

    # Stage 2: Conv2d (384 -> 384) + GELU
    C_in2 = C_out1
    C_out2 = C_in2  # 384
    H2 = H_out1
    W2 = W_out1
    H_out2 = (H2 - 3) // 2 + 1  # 20
    W_out2 = (W2 - 3) // 2 + 1  # T // 4
    y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=dtype, device=device)
    x2 = y1_gelu
    grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
    conv2d_3x3_stride2_pad1_kernel[grid2](
        x2, conv2d2_weight, conv2d2_bias, y2,
        B, C_in2, H2, W2, C_out2, H_out2, W_out2,
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        BLOCK_CO=64,
    )
    y2_gelu = torch.empty_like(y2)
    N2 = y2.numel()
    gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

    # Stage 3: Conv2d (384 -> 384) + GELU
    C_in3 = C_out2
    C_out3 = C_in3  # 384
    H3 = H_out2
    W3 = W_out2
    H_out3 = (H3 - 3) // 2 + 1  # 10
    W_out3 = (W3 - 3) // 2 + 1  # T // 8
    y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=dtype, device=device)
    x3 = y2_gelu
    grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
    conv2d_3x3_stride2_pad1_kernel[grid3](
        x3, conv2d3_weight, conv2d3_bias, y3,
        B, C_in3, H3, W3, C_out3, H_out3, W_out3,
        x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
        y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
        BLOCK_CO=64,
    )

    # Reshape: (B, channels, freq, time) -> (B, time, channels*freq)
    b, c, f, t = y3.size()  # B, 384, 10, T//8
    y3_perm = y3.permute(0, 3, 1, 2).contiguous()  # (B, T//8, 384, 10)
    K = c * f  # 3840
    X = y3_perm.view(B, t, K)  # (B, T//8, 3840)

    # Linear projection to d_model (no bias): output (B, T//8, 1024)
    M = 1024
    Y = torch.empty((B, t, M), dtype=dtype, device=device)
    # Ensure inputs are contiguous
    X_c = X.contiguous()
    W_c = conv_out_weight.contiguous()
    stride_xb, stride_xt, stride_xk = X_c.stride()
    stride_wm, stride_wk = W_c.stride()
    stride_yb, stride_yt, stride_ym = Y.stride()
    grid_linear = (B, t, triton.cdiv(M, 64))
    linear_no_bias_kernel[grid_linear](
        X_c, W_c, Y,
        B, t, K, M,
        stride_xb, stride_xt, stride_xk,
        stride_wm, stride_wk,
        stride_yb, stride_yt, stride_ym,
        BLOCK_M=64, BLOCK_K=256,
    )

    # Scale embeddings
    # Y is already computed; we need to add scaled positional embedding
    T_after = t  # time_after_conv from input, passed as axes["time_dim"] // 8
    # Slice positional_embedding to (T_after, d_model)
    pos_slice = positional_embedding[:T_after, :].contiguous()  # (T_after, 1024)
    # Launch Triton kernel to add scaled pos_emb
    scale = embed_scale
    grid_pos = (B, T_after, triton.cdiv(M, 64))
    add_scaled_pos_emb_kernel[grid_pos](
        Y, pos_slice, scale,
        B, T_after, M,
        Y.stride(0), Y.stride(1), Y.stride(2),
        pos_slice.stride(0), pos_slice.stride(1),
    )

    return Y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        return run(*args)


def run(*args):
    return ModelNew()(*args)
