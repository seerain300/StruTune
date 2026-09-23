import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1, NCHW layout
# X: (B, C_in, H, W) — input
# W: (C_out, C_in, 3, 3) — weights
# Bias: (C_out)
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
    h_out = tl.program_id(1)  # tile over H_out positions
    w_out = tl.program_id(2)  # tile over W_out positions

    # Compute base h_out and w_out scalar indices for this program
    h_out_idx = h_out
    w_out_idx = w_out

    # Output channel tile
    co_start = tl.program_id(3) * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ci in range(0, C_in):
        # For each kh,kw, compute corresponding input coordinates
        for kh in range(0, 3):
            in_h = h_out_idx * 2 + 1 - kh
            for kw in range(0, 3):
                in_w = w_out_idx * 2 + 1 - kw

                # Validity of input coordinates
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)

                # Base input pointers for this (b, ci, in_h, in_w)
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                # Load input (scalar) with mask; convert to float32 for accumulation
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                # Load weights for this (co_tile, ci, kh, kw), vector over BLOCK_CO
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # Accumulate
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y at (b, co_tile, h_out_idx, w_out_idx)
    y_ptrs = Y + pid_b * stride_yb + co_offsets * stride_yc + h_out_idx * stride_yh + w_out_idx * stride_yw
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
# X: (B, T, K) — flattened, passed as 1D contiguous
# W: (M, K) — weights
# Y: (B, T, M) — output
@triton.jit
def linear_no_bias_kernel(
    X_flat, W, Y_flat,
    B, T, K, M,
    BLOCK_M: tl.constexpr,  # tile over output channels M
    BLOCK_K: tl.constexpr   # tile over reduction K
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[t, k_offsets]
        x_ptrs = X_flat + t * K + k_offsets
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # shape (BLOCK_K,)

        # Load W[m_offsets, k_offsets] as (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * K + k_offsets[None, :]
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc[m] += sum_k (X[t, k] * W[m, k])
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y_flat + pid_b * (T * M) + t * M + m_offsets
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add scaled positional embedding to Y[b, t, m]
# Y: (B, T, M) — contiguous
# pos: (T, M) — positional embedding with dtype matching Y (bfloat16)
# scale: float32
@triton.jit
def add_scaled_pos_emb_kernel(Y, pos, scale, B, T, M, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t = pid_t
    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        y_ptrs = Y + pid_b * (T * M) + t * M + m_offsets
        y_val = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        pos_ptrs = pos + t * M + m_offsets
        pos_val = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        y_val = y_val + scale * pos_val
        tl.store(y_ptrs, y_val, mask=m_mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, C_in1, H1, W1 = input_features.shape
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T // 2
        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.bfloat16, device=input_features.device)

        # Ensure contiguous and correct strides
        x1_contig = x1.contiguous()
        x_strides = input_features.stride()
        w_strides = conv2d1_weight.stride()
        y1_strides = x1_contig.stride()
        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1_contig,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            w_strides[0], w_strides[1], w_strides[2], w_strides[3],
            y1_strides[0], y1_strides[1], y1_strides[2], y1_strides[3],
            BLOCK_CO=64,
        )

        # GELU1 (Triton kernel)
        N1 = x1_contig.numel()
        y1 = torch.empty_like(x1_contig, dtype=torch.float32)  # compute in fp32
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x1_contig, y1, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4
        x2 = y1  # use fp32 for conv to improve numerical stability

        x2_contig = x2.contiguous()
        y2_contig = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=input_features.device)

        x2_strides = x2_contig.stride()
        w2_strides = conv2d2_weight.stride()
        y2_strides = y2_contig.stride()

        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            x2_contig, conv2d2_weight, conv2d2_bias, y2_contig,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x2_strides[0], x2_strides[1], x2_strides[2], x2_strides[3],
            w2_strides[0], w2_strides[1], w2_strides[2], w2_strides[3],
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
            BLOCK_CO=64,
        )

        N2 = y2_contig.numel()
        y2 = torch.empty_like(y2_contig, dtype=torch.float32)
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2_contig, y2, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384) + GELU
        C_in3 = C_out2
        C_out3 = C_out2
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T // 8

        x3 = y2  # fp32
        x3_contig = x3.contiguous()
        y3_contig = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=input_features.device)

        x3_strides = x3_contig.stride()
        w3_strides = conv2d3_weight.stride()
        y3_strides = y3_contig.stride()

        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            x3_contig, conv2d3_weight, conv2d3_bias, y3_contig,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x3_strides[0], x3_strides[1], x3_strides[2], x3_strides[3],
            w3_strides[0], w3_strides[1], w3_strides[2], w3_strides[3],
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
            BLOCK_CO=64,
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = y3_contig.shape
        T_after_conv = t  # e.g., T // 8
        y3_flat = y3_contig.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T_after_conv, 3840) in fp32

        # Linear projection (no bias) via Triton: Y = X @ W^T
        # Ensure X_flat is contiguous 1D
        K = 3840
        M = conv_out_weight.shape[0]  # 1024
        X_flat = y3_flat.view(-1)  # (B * T_after_conv * K,)
        # Prepare W as (M, K)
        W = conv_out_weight.contiguous()  # (M=1024, K=3840)
        Y_flat = torch.empty((B * T_after_conv * M,), dtype=torch.float32, device=input_features.device)

        # Launch linear_no_bias_kernel
        grid = (B, T_after_conv, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid](
            X_flat, W, Y_flat,
            B, T_after_conv, K, M,
            BLOCK_M=128, BLOCK_K=256
        )

        # Reshape to (B, T_after_conv, M)
        Y = Y_flat.view(B, T_after_conv, M).to(torch.bfloat16)

        # Scale embeddings
        # Scale is float32; we'll perform scaling and add pos via Triton kernel
        pos = positional_embedding[:T_after_conv, :].to(torch.bfloat16).contiguous()  # (T_after_conv, 1024)
        # Triton kernel for adding scaled pos
        add_scaled_pos_emb_kernel[(B, T_after_conv)](
            Y, pos, float(embed_scale), B, T_after_conv, M, BLOCK_M=128
        )

        return Y


def run(*args):
    return ModelNew()(*args)
