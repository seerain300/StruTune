import math
import triton
import triton.language as tl


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K) with strides; W: (M, K) with strides; Y: (B, T, M) with strides
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduction over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[t, k_offsets]
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets] as a matrix (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # acc += sum_k W[m,k] * X[t,k]
        for kk in range(0, BLOCK_K):
            valid_kk = k_start + kk < K
            x_scalar = tl.load(X + pid_b * stride_xb + t * stride_xt + (k_start + kk) * stride_xk, mask=valid_kk, other=0.0).to(tl.float32)
            w_col = tl.load(W + m_offsets * stride_wm + (k_start + kk) * stride_wk, mask=m_mask, other=0.0).to(tl.float32)
            acc += x_scalar * w_col

    # Store
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise scaling + add sliced positional embedding
# X: (B, T, M); PE: (S, M) where S <= max_source_positions; Y: (B, T, M)
@triton.jit
def scale_add_pos_emb_kernel(X, PE, Y, B, T, M, S, scale, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Load X element
    x_ptrs = X + pid_b * stride_xb + t * stride_xt + m_offsets * stride_xm
    x_val = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load positional embedding row index = t (since we slice based on T_after_conv)
    # Note: we slice positional_embedding up to S in host code; here we assume T <= S (which is true).
    pe_ptrs = PE + t * stride_pes + m_offsets * stride_pem
    # t is always < T, and in these workloads T <= S, so always valid
    pe_val = tl.load(pe_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    y_val = x_val * scale + pe_val
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, y_val, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight,
                positional_embedding,
                embed_scale: float):
        """
        input_features: (B, 1, 80, T) bfloat16
        conv2d1_weight: (384, 1, 3, 3) bfloat16
        conv2d1_bias: (384) bfloat16
        conv2d2_weight: (384, 384, 3, 3) bfloat16
        conv2d2_bias: (384) bfloat16
        conv2d3_weight: (384, 384, 3, 3) bfloat16
        conv2d3_bias: (384) bfloat16
        conv_out_weight: (1024, 3840) bfloat16
        positional_embedding: (1500, 1024) bfloat16
        embed_scale: float (sqrt(1024) = 32.0)
        Returns: (B, T//8, 1024) bfloat16
        """

        B, C_in, H1, W1 = input_features.shape
        device = input_features.device
        dtype = input_features.dtype

        # Stage 1: Conv2d (1 -> 384) stride=2, padding=1
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # Stage 2: Conv2d (384 -> 384) stride=2, padding=1
        x2 = F.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        # Stage 3: Conv2d (384 -> 384) stride=2, padding=1
        x3 = F.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)  # (B, 384, 10, T//8)

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10) -> (B, T//8, 3840)
        Bsz, _, Ff, Tafter = x3.shape
        X_lin = x3.permute(0, 3, 1, 2).contiguous().view(Bsz, Tafter, 384 * 10)

        # Linear projection without bias: (B, Tafter, 3840) @ (1024, 3840)^T -> (B, Tafter, 1024)
        M = 1024  # d_model
        K = 3840
        X_lin = X_lin.contiguous()  # (B, Tafter, K)
        conv_out_weight_T = conv_out_weight.t().contiguous()  # (K, M)
        Y_lin = torch.empty((Bsz, Tafter, M), dtype=dtype, device=device)

        # Strides
        stride_xb, stride_xt, stride_xk = X_lin.stride()  # (Tafter, K)
        stride_wk, stride_wm = conv_out_weight_T.stride()  # (M, K)
        stride_yb, stride_yt, stride_ym = Y_lin.stride()

        # Launch Triton GEMM-like kernel
        BLOCK_M = 128
        BLOCK_K = 128
        grid = (Bsz, Tafter, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            X_lin, conv_out_weight_T, Y_lin,
            Bsz, Tafter, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wk, stride_wm,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
        )

        # Scale by embed_scale and add positional embedding
        # Y_lin shape: (B, Tafter, M)
        # positional_embedding: (1500, M), dtype=bfloat16
        Bsz, Tafter, M = Y_lin.shape
        S = Tafter  # we only need the first Tafter rows of positional_embedding
        Y_scaled = torch.empty_like(Y_lin)

        # Strides for Y and embedding (assumed contiguous)
        stride_yb, stride_yt, stride_ym = Y_scaled.stride()  # (Tafter, M)
        # We will pass a sliced embedding of shape (S, M) into the kernel; positional_embedding is (1500, M)
        # Ensure embedding is contiguous
        emb_slice = positional_embedding[:S, :].contiguous()  # (S, M)

        stride_pes, stride_pem = emb_slice.stride()  # (M, S)

        # Launch elementwise kernel
        BLOCK_M = 128
        grid = (Bsz, Tafter, triton.cdiv(M, BLOCK_M))
        scale_add_pos_emb_kernel[grid](
            Y_lin, emb_slice, Y_scaled,
            Bsz, Tafter, M, S, embed_scale,
            BLOCK_M=BLOCK_M
        )

        return Y_scaled


def run(*args):
    return ModelNew()(*args)
