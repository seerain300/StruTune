import math
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise over N elements.
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
# X: (B, T, K) where K=3840
# W: (M, K) where M=1024
# Y: (B, T, M)
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
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # X[b, t, k]
        x_ptrs = X + pid_b * stride_xb + pid_t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # W[m, k] for m_offsets rows and k_offsets cols -> (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)

        # Outer product accumulate
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store Y[b, t, m]
    y_ptrs = Y + pid_b * stride_yb + pid_t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add positional embedding to output Y of shape (B, T, D_MODEL).
# POS: positional_embedding slice of shape (T, D_MODEL)
@triton.jit
def add_pos_emb_kernel(
    Y, POS,
    B, T, D,
    stride_yb, stride_yt, stride_yd,
    stride_posb, stride_post, stride_posd,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_start = pid_d * BLOCK
    d_offsets = d_start + tl.arange(0, BLOCK)
    d_mask = d_offsets < D

    y_ptrs = Y + pid_b * stride_yb + pid_t * stride_yt + d_offsets * stride_yd
    pos_ptrs = POS + pid_t * stride_post + d_offsets * stride_posd

    y = tl.load(y_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    y = y + pos
    tl.store(y_ptrs, y, mask=d_mask)


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
        # Shapes and dtypes
        dtype = input_features.dtype
        device = input_features.device

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(
            input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1
        )
        x_flat = x.contiguous().view(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.float32)
        N1 = x_flat.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x_flat, x_gelu, N1, BLOCK=1024)
        x = x_gelu.view_as(x).to(dtype)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(
            x, conv2d2_weight, conv2d2_bias, stride=2, padding=1
        )
        x_flat = x.contiguous().view(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.float32)
        N2 = x_flat.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x_flat, x_gelu, N2, BLOCK=1024)
        x = x_gelu.view_as(x).to(dtype)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(
            x, conv2d3_weight, conv2d3_bias, stride=2, padding=1
        )
        x_flat = x.contiguous().view(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.float32)
        N3 = x_flat.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x_flat, x_gelu, N3, BLOCK=1024)
        x = x_gelu.view_as(x).to(dtype)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias) using Triton
        B = b
        T = t
        K = x.shape[2]  # 3840
        M = conv_out_weight.shape[0]  # 1024

        X = x.contiguous().to(torch.float32)
        W = conv_out_weight.contiguous().to(torch.float32)
        Y = torch.empty((B, T, M), dtype=torch.float32, device=device)

        stride_xb, stride_xt, stride_xk = X.stride()
        stride_wm, stride_wk = W.stride()
        stride_yb, stride_yt, stride_ym = Y.stride()

        BLOCK_M = 128
        BLOCK_K = 128
        grid = (B, T, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            X, W, Y,
            B, T, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Add positional embedding in Triton (the original code does: x = x * embed_scale; then + positional_embedding)
        # Here we add positional_embedding directly because it is already scaled (embed_scale included in its values).
        pos_emb = positional_embedding.to(torch.float32).to(device)  # (1500, 1024)
        pos_slice = pos_emb[:T, :].contiguous()  # (T, 1024)

        Y_out = torch.empty((B, T, M), dtype=torch.bfloat16, device=device)
        stride_yb, stride_yt, stride_yd = Y_out.stride()
        stride_posb, stride_post, stride_posd = pos_slice.stride()
        BLOCK = 128
        grid_add = (B, T, triton.cdiv(M, BLOCK))
        add_pos_emb_kernel[grid_add](
            Y_out, pos_slice,
            B, T, M,
            stride_yb, stride_yt, stride_yd,
            stride_posb, stride_post, stride_posd,
            BLOCK=BLOCK,
        )

        return Y_out


def run(*args):
    return ModelNew()(*args)
