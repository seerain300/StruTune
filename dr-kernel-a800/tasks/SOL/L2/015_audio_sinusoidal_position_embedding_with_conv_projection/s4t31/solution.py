import math
import triton
import triton.language as tl


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
# X: (B, T, K) with strides
# W: (M, K) with strides
# Y: (B, T, M) with strides
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # time index
    pid_m = tl.program_id(2)  # tile over output channels M

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] as a (BLOCK_M, BLOCK_K) tile
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store result
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale + add positional embedding
# Y: (B, T, M) — input
# POS: (T, M) — positional embedding (we slice only T rows)
# SCALE: float scalar
@triton.jit
def add_scaled_pos_emb_kernel(
    Y, POS, OUT,
    B, T, M,
    stride_yb, stride_yt, stride_ym,
    stride_pos_t, stride_pos_m,
    SCALE: tl.float32,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Load Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    y_vals = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load POS[t, m_offsets]
    pos_ptrs = POS + t * stride_pos_t + m_offsets * stride_pos_m
    pos_vals = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Scale and add
    y_vals = y_vals + pos_vals * SCALE

    # Store to OUT
    out_ptrs = OUT + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(out_ptrs, y_vals, mask=m_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
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
        # Ensure contiguity for Triton kernels (dtype stays bfloat16 throughout)
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        positional_embedding = positional_embedding.contiguous()

        B = input_features.shape[0]
        H1 = input_features.shape[2]
        W1 = input_features.shape[3]

        # Stage 1: Conv2d (1 -> 384 channels), stride=2, padding=1
        # Outputs: (B, 384, H_out1, W_out1)
        C_out1 = 384
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T // 2

        y1 = torch.nn.functional.conv2d(
            input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1
        )
        y1 = y1.contiguous()

        # GELU1 via Triton
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384)
        C_in2 = C_out1
        H2 = H_out1
        W2 = W_out1
        C_out2 = C_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4

        y2 = torch.nn.functional.conv2d(
            y1_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1
        )
        y2 = y2.contiguous()

        # GELU2 via Triton
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384)
        C_in3 = C_out2
        H3 = H_out2
        W3 = W_out2
        C_out3 = C_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T // 8

        y3 = torch.nn.functional.conv2d(
            y2_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1
        )
        y3 = y3.contiguous()

        # Reshape to (B, T_after_conv, 384)
        B_final = B
        T_after_conv = W_out3
        C_out_dim = C_out3  # 384

        x_lin = y3.permute(0, 3, 1, 2).contiguous().view(B_final, T_after_conv, C_out_dim).contiguous()

        # Linear projection (no bias) via Triton
        # X: (B, T_after_conv, K=3840), W: (M=1024, K=3840)
        X = x_lin.contiguous().to(torch.float32)  # compute in fp32 for stability
        W = conv_out_weight.contiguous().to(torch.float32)

        Y = torch.empty((B_final, T_after_conv, C_out_dim), dtype=torch.float32, device=input_features.device)

        BLOCK_M = 128
        BLOCK_K = 128
        grid_linear = (B_final, T_after_conv, triton.cdiv(C_out_dim, BLOCK_M))
        linear_no_bias_kernel[grid_linear](
            X, W, Y,
            B_final, T_after_conv, 3840, C_out_dim,
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale (sqrt(1024) = 32.0)
        Y_scaled = Y * embed_scale

        # Add positional embedding (first T_after_conv rows), dtype fp32
        POS = positional_embedding[:T_after_conv, :].contiguous().to(torch.float32)
        OUT = torch.empty_like(Y_scaled, dtype=torch.float32, device=input_features.device)

        # Launch Triton add kernel
        grid_pos = (B_final, T_after_conv, triton.cdiv(C_out_dim, BLOCK_M))
        add_scaled_pos_emb_kernel[grid_pos](
            Y_scaled, POS, OUT,
            B_final, T_after_conv, C_out_dim,
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            POS.stride(0), POS.stride(1),
            float(embed_scale),
            BLOCK_M=BLOCK_M,
        )

        # Cast back to bfloat16 to match original pipeline
        out = OUT.to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
