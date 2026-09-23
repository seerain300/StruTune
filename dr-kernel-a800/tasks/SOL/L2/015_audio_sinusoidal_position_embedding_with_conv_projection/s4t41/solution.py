import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# Input: X (B, C_in, H, W) — supports non-contiguous via strides
# Weights: W (C_out, C_in, 3, 3) — supports non-contiguous via strides
# Bias: BIAS (C_out) — contiguous
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)

    # Decode h_out, w_out from pid_hw
    hw_total = H_out * W_out
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Reduction over input channels and 3x3 kernel
    # Use masks for X loads to avoid OOB due to padding
    for ci in range(0, C_in):
        # For each kernel tap
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = 2 * h_out + 1 - kh  # due to padding=1
                in_w = 2 * w_out + 1 - kw
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)

                # Load x tile for all CO offsets
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                # Load corresponding weights vector for all CO offsets
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # FMA
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store result
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

        # Accumulate: acc += sum_k W[m,k] * X[t,k]
        # Manually FMA across the BLOCK_K dimension
        # Unroll multiply-add over BLOCK_K
        for kk in range(0, BLOCK_K):
            # mask on kk for tail
            valid_kk = k_start + kk < K
            # For this kk, load x_scalar
            x_scalar = tl.load(X + pid_b * stride_xb + t * stride_xt + (k_start + kk) * stride_xk, mask=valid_kk, other=0.0).to(tl.float32)
            # Load corresponding column of W
            w_col_ptrs = W + m_offsets * stride_wm + (k_start + kk) * stride_wk
            w_col = tl.load(w_col_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += x_scalar * w_col

    # Store
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise add scaled pos embedding
# Input: X (B, T, M) — contiguous or strided
# Embedding: PE (1500, M) — contiguous or strided (we will pass a slice of 1500 rows)
@triton.jit
def add_scaled_pos_emb_kernel(X, PE, Y, B, T, M, scale, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Load X[b, t, m_offsets]
    x_ptrs = X + pid_b * stride_xb + t * stride_xt + m_offsets * stride_xm
    x = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load scaled pos embedding row for t (index row = t, since we slice to T rows)
    # Note: Pass PE as (T, M) slice; we assume caller passed a valid slice.
    pe_ptrs = PE + t * stride_pert + m_offsets * stride_pem
    pe = tl.load(pe_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    y = x + pe * scale

    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, y, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight,
                positional_embedding, embed_scale):
        # Ensure dtypes: compute in float32 for accumulation, store in bfloat16 as per get_inputs
        # We will pass tensors as-is; Triton kernels will load and cast as needed.
        B, C_in, H, W_in = input_features.shape
        C_out1 = conv2d1_weight.shape[0]
        C_out2 = conv2d2_weight.shape[0]
        C_out3 = conv2d3_weight.shape[0]

        # Stage 1: Conv2d (1 -> 384), stride=2, padding=1
        H_out1 = (H - 3) // 2 + 1
        W_out1 = (W_in - 3) // 2 + 1
        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=input_features.device)
        conv2d_3x3_stride2_pad1_kernel[(B, H_out1 * W_out1, triton.cdiv(C_out1, 64))](input_features, conv2d1_weight, conv2d1_bias, y1,
                                                                                   B, 1, H, W_in, C_out1, H_out1, W_out1,
                                                                                   input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
                                                                                   conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
                                                                                   y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
                                                                                   BLOCK_CO=64)
        # GELU after conv1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384), stride=2, padding=1
        y2 = torch.empty((B, C_out2, (H_out1 - 3) // 2 + 1, (W_out1 - 3) // 2 + 1),
                         dtype=torch.float32, device=input_features.device)
        conv2d_3x3_stride2_pad1_kernel[(B, ((H_out1 - 3) // 2 + 1) * ((W_out1 - 3) // 2 + 1), triton.cdiv(C_out2, 64))](y1_gelu, conv2d2_weight, conv2d2_bias, y2,
                                                                                     B, C_out1, H_out1, W_out1, C_out2,
                                                                                     y1_gelu.stride(0), y1_gelu.stride(1), y1_gelu.stride(2), y1_gelu.stride(3),
                                                                                     conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
                                                                                     y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
                                                                                     BLOCK_CO=64)
        # GELU after conv2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384), stride=2, padding=1
        H3 = (H_out1 - 3) // 2 + 1
        W3 = (W_out1 - 3) // 2 + 1
        H_out3 = (H3 - 3) // 2 + 1
        W_out3 = (W3 - 3) // 2 + 1
        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=input_features.device)
        conv2d_3x3_stride2_pad1_kernel[(B, H_out3 * W_out3, triton.cdiv(C_out3, 64))](y2_gelu, conv2d3_weight, conv2d3_bias, y3,
                                                                                   B, C_out2, H3, W3, C_out3, H_out3, W_out3,
                                                                                   y2_gelu.stride(0), y2_gelu.stride(1), y2_gelu.stride(2), y2_gelu.stride(3),
                                                                                   conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
                                                                                   y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
                                                                                   BLOCK_CO=64)
        # GELU after conv3
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Flatten to (B, T_after_conv, K=3840)
        B, C_out3, H_out3, W_out3 = y3_gelu.shape
        T_after_conv = W_out3  # time is last dimension
        # The original code reshapes to (B, time_after_conv, C_out_dim) where C_out_dim = 10*96=960? No, original: (B, time_after_conv, 1024). The code says conv_out_dim=3840 and after convs we have 384 channels, time T//8, so reshape (B, T//8, 384*10)= (B, T//8, 3840). Let's clarify:
        # We have (B, 384, 10, T//8). 384*10=3840. So we should have K=3840.
        # However, conv_out_weight is (1024, 3840). So the code performs linear on 3840 features to produce 1024. That means the "time" after conv is not the time; it's a feature dimension of size 3840. To match original, we need to reshape as (B, T_after_conv, K=3840), but the original T_after_conv is the time after the 3rd conv, which is T//8. We don't have it; the original code uses time_dim and constructs time_after_conv. In their code, they use t for time and then do (B, t, 384*10). They set conv_out_dim=3840. So our y3_gelu has shape (B, 384, 10, T//8). Let's proceed by computing (B, T//8, 384*10) and then linear to 1024. Wait, the provided conv_out_weight is (1024, 3840), so they expect input K=3840, not 3840*10. This implies the original code intended the convs output 3840 features, not 384*10. To be consistent, we will reshape to (B, T_after_conv, 3840) where T_after_conv is the last dimension of y3_gelu, which is T//8. Then linear to 1024.

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10) would be incorrect for the given conv_out_weight. The original code sets conv_out_dim=3840, and conv_out_weight is (1024, 3840). Therefore, the model expects the input to linear to have 3840 features. In the original, the comment says "conv_out_weight": xavier(d_model, conv_out_dim), where d_model=1024 and conv_out_dim=3840, so the output is 1024. The convs produce (B, 384, 10, T//8); to make 3840 features, they must concatenate or combine the 10 channels into a single feature vector of size 3840. However, since we cannot change get_inputs (it returns conv_out_weight (1024, 3840)), we will assume the model’s intent: after convs, they somehow reduce or combine to K=3840 features, then linear to 1024. To keep compatibility, we'll define K as 3840. If the conv output has fewer features, we can pad with zeros; but per the provided initialization, conv_out_dim=3840. Therefore, we will use K=3840.

        # But the actual conv output has 384 channels; 384*10=3840. We can extract features from the 384 channels across 10 and T//8, but we need exactly 3840. A straightforward way is to take the last 10 channels and flatten across spatial. However, they say conv_out_dim=3840, which corresponds to 384*10. The code likely assumes that the conv output contains 3840 features implicitly; given conv_out_weight is (1024, 3840), we'll proceed by concatenating across channels. We can reshape y3_gelu to (B, C_out3, 10, T//8) -> (B, T//8, C_out3*10) if C_out3*10==3840. Given C_out3=384 and 10*384=3840, this matches.

        # Reshape to (B, T_after_conv, 3840)
        y3_gelu = y3_gelu.permute(0, 3, 1, 2).contiguous()  # (B, T//8, 384, 10)
        B, T_after_conv, C3, ten = y3_gelu.shape
        # Assume conv_out_dim is 384*10=3840 from get_inputs. Let's assert it by combining channels.
        K = C3 * ten  # 384 * 10 = 3840
        X_lin = y3_gelu.reshape(B, T_after_conv, K)  # (B, T_after_conv, 3840)

        # Linear projection (no bias): Y = X @ W^T, W: (1024, 3840)
        Y_lin = torch.empty((B, T_after_conv, 1024), dtype=torch.float32, device=input_features.device)
        linear_no_bias_kernel[(B, T_after_conv, triton.cdiv(1024, 64))](X_lin, conv_out_weight, Y_lin,
                                                                      B, T_after_conv, K, 1024,
                                                                      X_lin.stride(0), X_lin.stride(1), X_lin.stride(2),
                                                                      conv_out_weight.stride(0), conv_out_weight.stride(1),
                                                                      Y_lin.stride(0), Y_lin.stride(1), Y_lin.stride(2),
                                                                      BLOCK_M=64, BLOCK_K=64)

        # Scale by embed_scale (sqrt(1024) = 32) and add positional embedding
        # Positional embedding shape: (1500, 1024). We only need rows [0 : T_after_conv].
        # Ensure embed_scale is float
        scale = float(embed_scale)
        # We need Y_lin to be (B, T_after_conv, 1024). We can add_scaled_pos_emb_kernel. But we must have X for the kernel as (B, T, M). Since we already have Y_lin, we can call the kernel with X=Y_lin (we will read and add), and create an output tensor. Note: This kernel expects X input before addition, but we'll pass Y_lin as input and output the same, since we compute Y = X + scale*PE. Triton will read X and write Y.
        Y_final = torch.empty_like(Y_lin)
        add_scaled_pos_emb_kernel[(B, T_after_conv, triton.cdiv(1024, 64))](
            Y_lin, positional_embedding, Y_final, B, T_after_conv, 1024, scale,
            64  # BLOCK_M
        )

        return Y_final


def run(*args):
    return ModelNew()(*args)
