import math
import torch
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise
# Applies GELU to input X and writes to Y. Works on any contiguous 1D buffer.
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
# X: (B, T, K) contiguous, row-major along T
# W: (M, K) contiguous, row-major along K
# Y: (B, T, M) contiguous, row-major along M
# We tile over M (BLOCK_M) and loop over K in chunks (BLOCK_K).
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

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets] as vector
        x_ptrs = X + pid_b * stride_xb + pid_t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets] as matrix (BLOCK_M x BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k x_vec[k] * w_mat[:, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

        k_start += BLOCK_K

    # Store acc to Y
    y_ptrs = Y + pid_b * stride_yb + pid_t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise scale (multiply by scalar)
@triton.jit
def scale_kernel(X, Y, N, S, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * S
    tl.store(Y + offsets, y, mask=mask)


# Triton kernel: Add positional embedding across last dimension
# Z: (B, T, M), P: (M, D) — positional embedding
# We add P[:, d] to all rows along channel dimension M at position d. Here we implement simple elementwise
# broadcasting since D=M in provided setup. For general, adapt to slice rows based on seq_len if needed.
@triton.jit
def add_pos_emb_kernel(Z, P, B, T, M, D, S, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Iterate over rows m=0..M-1, add P[m, d_offsets] * S to Z[b, t, d_offsets]
    # We load Z[b, t, d_offsets], add scaled P[:, d_offsets], store back.
    z_ptrs = Z + pid_b * (T * M) + pid_t * M + d_offsets
    z_vals = tl.load(z_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Iterate over m
    m = 0
    while m < M:
        p_ptrs = P + m * D + d_offsets
        p_vals = tl.load(p_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        p_vals = p_vals * S
        z_vals = z_vals + p_vals
        m += 1

    tl.store(Z + pid_b * (T * M) + pid_t * M + d_offsets, z_vals, mask=d_mask)


def triton_gelu(x):
    # Launch GELU Triton kernel on x (1D flattened), returns y of same shape
    x_flat = x.reshape(-1)
    y_flat = torch.empty_like(x_flat, dtype=torch.float32)
    N = x_flat.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    gelu_tanh_kernel[grid](x_flat, y_flat, N, BLOCK=BLOCK)
    return y_flat.view_as(x).to(x.dtype)


def triton_linear_no_bias(X, W):
    # X: (B, T, K), W: (M, K), returns Y: (B, T, M)
    X_ = X.contiguous()
    W_ = W.contiguous()
    B, T, K = X_.shape
    M, K_w = W_.shape
    assert K == K_w, "K mismatch between X and W"
    Y = torch.empty((B, T, M), dtype=torch.float32, device=X_.device)

    stride_xb, stride_xt, stride_xk = X_.stride()
    stride_wm, stride_wk = W_.stride()
    stride_yb, stride_yt, stride_ym = Y.stride()

    BLOCK_M = 64
    BLOCK_K = 128
    grid = (B, T, triton.cdiv(M, BLOCK_M))
    linear_no_bias_kernel[grid](
        X_, W_, Y,
        B, T, K, M,
        stride_xb, stride_xt, stride_xk,
        stride_wm, stride_wk,
        stride_yb, stride_yt, stride_ym,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return Y.to(X.dtype)


def triton_scale(X, scale):
    # Scale X elementwise by 'scale' using Triton
    X_ = X.contiguous()
    N = X_.numel()
    Y = torch.empty_like(X_, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    scale_kernel[grid](X_.view(-1), Y.view(-1), N, scale, BLOCK=BLOCK)
    return Y.to(X.dtype)


def triton_add_pos_emb(Z, pos_emb, embed_scale):
    # Z: (B, T, M), pos_emb: (M, D)
    # We assume M == D for the provided setup (M=1024, D=1024). If not, fallback logic can be added.
    B, T, M = Z.shape
    D = pos_emb.shape[1]
    assert M == D, "M must equal D for positional embedding addition"
    Z_ = Z.contiguous()
    P = pos_emb.contiguous()
    BLOCK_D = 128
    grid = (B, T, triton.cdiv(M, BLOCK_D))
    add_pos_emb_kernel[grid](
        Z_.view(-1, M), P, B, T, M, D, embed_scale, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2
    )
    return Z_.to(Z.dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU (Triton GELU)
        x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x1 = triton_gelu(x1)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = torch.nn.functional.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = triton_gelu(x2)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = torch.nn.functional.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = triton_gelu(x3)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x3.size()
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias) — Triton kernel
        y = triton_linear_no_bias(x3, conv_out_weight)

        # Scale embeddings by embed_scale (sqrt(d_model) = 32.0)
        y = triton_scale(y, embed_scale)

        # Add positional embeddings — Triton kernel
        y = triton_add_pos_emb(y, positional_embedding, embed_scale)

        return y


def run(*args):
    return ModelNew()(*args)
