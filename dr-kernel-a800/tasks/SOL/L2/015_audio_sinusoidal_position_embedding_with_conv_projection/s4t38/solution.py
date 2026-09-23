import math
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise
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
# X: (B, T, K) where K=3840, float32 (accumulation)
# W: (M, K) where M=1024, conv_out_weight
# Y: (B, T, M) float32, later cast to bfloat16
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,   # tile over M (output channels), e.g., 128
    BLOCK_K: tl.constexpr    # tile over K (reduction), e.g., 128
):
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # time index
    pid_m = tl.program_id(2)  # tile index over M

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

        # Load X[b, t, k_offsets] -> shape (BLOCK_K,)
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # Accumulate: acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store acc to Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale and add positional embedding
# X: (B, T, M) float32
# POS: (T, M) float32 (embed_scale * positional_embedding[:T, :])
@triton.jit
def scale_add_pos_emb_kernel(X, POS, Y, B, T, M,
                              stride_xb, stride_xt, stride_xm,
                              stride_posb, stride_posti, stride_posm,
                              stride_yb, stride_yt, stride_ym,
                              SCALE: tl.constexpr,
                              BLOCK: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK
    m_offsets = m_start + tl.arange(0, BLOCK)
    m_mask = m_offsets < M

    # Load X[b, t, m_offsets]
    x_ptrs = X + b * stride_xb + t * stride_xt + m_offsets * stride_xm
    x_vec = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load POS[t, m_offsets] (we assume POS is of shape (T, M) and contiguous)
    pos_ptrs = POS + t * stride_posti + m_offsets * stride_posm
    pos_vec = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Compute Y = SCALE * X + POS
    y_vec = SCALE * x_vec + pos_vec

    # Store
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, y_vec, mask=m_mask)


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
        # Stage 1: Conv2d (1 -> 384) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = x.contiguous()
        y1 = torch.empty_like(x)
        N1 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x, y1, N1, BLOCK=1024)
        x = y1  # bfloat16

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = x.contiguous()
        y2 = torch.empty_like(x)
        N2 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x, y2, N2, BLOCK=1024)
        x = y2  # bfloat16

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = x.contiguous()
        y3 = torch.empty_like(x)
        N3 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x, y3, N3, BLOCK=1024)
        x = y3  # bfloat16

        # Reshape: (B, C, H, W) -> (B, T_after, K=384*10)
        B, C, H, W = x.size()
        K = C * 10  # 384 * 10 = 3840
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, K)  # (B, T_after_conv, 3840)

        # Linear projection (no bias) in Triton
        M = conv_out_weight.shape[0]  # 1024
        # Ensure contiguous float32 for input X and weights
        Xc = x.to(torch.float32).contiguous()  # (B, T, K)
        Wc = conv_out_weight.contiguous()      # (M, K)
        Y = torch.empty((B, x.shape[1], M), dtype=torch.float32, device=x.device)

        # Launch Triton linear_no_bias_kernel
        BLOCK_M = 128
        BLOCK_K = 128
        T_after = x.shape[1]
        grid = (B, T_after, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            Xc, Wc, Y,
            B, T_after, K, M,
            Xc.stride(0), Xc.stride(1), Xc.stride(2),
            Wc.stride(0), Wc.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale (float) and add positional embedding in Triton
        # positional_embedding is (1500, 1024) in float32 from get_inputs; we slice [:T_after, :]
        POS = positional_embedding.to(torch.float32)  # keep as fp32 for stable math
        POS = POS[:T_after, :].contiguous()          # (T_after, 1024)
        Y_scaled = torch.empty_like(Y, dtype=torch.float32, device=Y.device)
        scale_add_pos_emb_kernel[(B, T_after, triton.cdiv(M, 128))](Y, POS, Y_scaled, B, T_after, M,
                                                                   Y.stride(0), Y.stride(1), Y.stride(2),
                                                                   POS.stride(0), POS.stride(1), POS.stride(1),
                                                                   Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
                                                                   SCALE=float(embed_scale),
                                                                   BLOCK=128)
        # Convert to bfloat16 for output (to match typical bfloat16 usage in get_inputs)
        out = Y_scaled.to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
