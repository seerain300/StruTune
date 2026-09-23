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
# X: (B, T, K), W: (M, K), Y: (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,   # tile size over M
    BLOCK_K: tl.constexpr    # tile size over reduction dimension
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

    # Store results Y[b, t, m_offsets] in bfloat16
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add scaled positional embedding to Y
# Y: (B, T, D), POS: (T, D) — note: we pass sliced POS[:T, :] from host
@triton.jit
def add_scaled_pos_emb_kernel(Y, POS, B, T, D, scale, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t_start = pid_t * BLOCK_T
    d_start = pid_d * BLOCK_D

    t_offsets = t_start + tl.arange(0, BLOCK_T)
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    t_mask = t_offsets < T
    d_mask = d_offsets < D

    # Load Y[b, t_offsets, d_offsets]
    y_ptrs = Y + b * stride_yb + t_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    y_mask = t_mask[:, None] & d_mask[None, :]
    y_mat = tl.load(y_ptrs, mask=y_mask, other=0.0).to(tl.float32)

    # Load POS[t_offsets, d_offsets]
    pos_ptrs = POS + t_offsets[:, None] * stride_pt + d_offsets[None, :] * stride_pd
    pos_mask = t_mask[:, None] & d_mask[None, :]
    pos_mat = tl.load(pos_ptrs, mask=pos_mask, other=0.0).to(tl.float32)

    y_mat += scale * pos_mat

    # Store back
    tl.store(y_ptrs, y_mat, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args are: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale

        # Extract tensors (assuming 8 positional arguments as per get_inputs)
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # shape (d_model=1024, conv_out_dim=3840)
        positional_embedding = args[8]  # shape (max_source_positions=1500, d_model=1024), bfloat16
        embed_scale = float(args[9])  # scalar float

        device = input_features.device
        dtype = input_features.dtype  # bfloat16 in this benchmark

        # Stage 1: Conv2d (1 -> 384) + GELU
        x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x1 = x1.contiguous()
        # GELU in Triton
        x1_gelu = torch.empty_like(x1)
        N1 = x1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 4096),)](x1, x1_gelu, N1, BLOCK=4096)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x2 = torch.nn.functional.conv2d(x1_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = x2.contiguous()
        x2_gelu = torch.empty_like(x2)
        N2 = x2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 4096),)](x2, x2_gelu, N2, BLOCK=4096)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x3 = torch.nn.functional.conv2d(x2_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = x3.contiguous()
        x3_gelu = torch.empty_like(x3)
        N3 = x3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 4096),)](x3, x3_gelu, N3, BLOCK=4096)

        # Reshape: (B, C, F, T) -> (B, T, C*F)
        B, C, F, T = x3_gelu.shape
        x3_gelu = x3_gelu.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        C_out_dim = C * F  # 384 * 10 = 3840

        # Linear projection to d_model (no bias) using Triton
        # X: (B, T, K=3840), W: (M=1024, K=3840), Y: (B, T, 1024)
        X = x3_gelu.view(B, T, C_out_dim)
        W = conv_out_weight  # (1024, 3840)
        Y = torch.empty((B, T, 1024), dtype=dtype, device=device)

        # Launch Triton linear kernel with tiling
        BLOCK_M = 64  # output channels tile
        BLOCK_K = 64  # reduction tile
        grid = (B, triton.cdiv(T, 1), triton.cdiv(1024, BLOCK_M))
        linear_no_bias_kernel[grid](
            X, W, Y,
            B, T, C_out_dim, 1024,
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
        )

        # Scale embeddings (store in a temporary to avoid in-place issues)
        Y_scaled = torch.empty_like(Y)
        # Y is (B, T, 1024), positional_embedding is (1500, 1024). Slice [:T, :].
        # Triton kernel expects POS of shape (T, D). We pass sliced POS from host.
        # To ensure correctness, we compute T = Y.shape[1] on host; slice positional_embedding accordingly.
        D = Y.shape[2]
        T_eff = Y.shape[1]
        POS = positional_embedding[:T_eff, :].contiguous()  # (T, D), bfloat16

        add_scaled_pos_emb_kernel[(B, triton.cdiv(T_eff, 128), triton.cdiv(D, 128))](Y_scaled, POS, B, T_eff, D, embed_scale, 128, 128)

        return Y_scaled


def run(*args):
    return ModelNew()(*args)
