import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise over N elements.
# X: input flattened tensor, Y: output tensor, N: total number of elements.
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


# Triton kernel: Linear projection (no bias) Y = X @ W^T for X: (B, T, K), W: (M, K), Y: (B, T, M)
# We tile over M (output channels) and loop over K in chunks.
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

        # Load X[b, t, k_offsets] -> vector of length BLOCK_K
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


# Triton kernel: Add scaled positional embedding to Y.
# Y: (B, T, M), POS: (T, M), SCALE: scalar
@triton.jit
def add_scaled_pos_emb_kernel(
    Y, POS, SCALE,
    B, T, M,
    stride_yb, stride_yt, stride_ym,
    stride_pos_t, stride_pos_m
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * 64
    m_offsets = m_start + tl.arange(0, 64)
    m_mask = m_offsets < M

    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    y_vals = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    pos_ptrs = POS + t * stride_pos_t + m_offsets * stride_pos_m
    pos_vals = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    y_vals = y_vals + SCALE * pos_vals

    tl.store(y_ptrs, y_vals, mask=m_mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    # Elementwise GELU (tanh approx) on x
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    gelu_tanh_kernel[grid](x, y, N, BLOCK=BLOCK, num_warps=4, num_stages=2)
    return y


def triton_linear_no_bias(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: (B, T, K), w: (M, K)
    B, T, K = x.shape
    M = w.shape[0]
    x_c = x.contiguous()
    w_c = w.contiguous()
    # Compute in fp32
    y = torch.empty((B, T, M), dtype=torch.float32, device=x.device)
    stride_xb, stride_xt, stride_xk = x_c.stride()
    stride_wm, stride_wk = w_c.stride()
    stride_yb, stride_yt, stride_ym = y.stride()

    BLOCK_M = 64
    BLOCK_K = 64
    grid = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(M, BLOCK_M))
    linear_no_bias_kernel[grid](
        x_c, w_c, y,
        B, T, K, M,
        stride_xb, stride_xt, stride_xk,
        stride_wm, stride_wk,
        stride_yb, stride_yt, stride_ym,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    # Cast back to original dtype
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


def triton_add_scaled_pos_emb(y: torch.Tensor, pos_slice: torch.Tensor, scale: float) -> torch.Tensor:
    # y: (B, T, M), pos_slice: (T, M)
    B, T, M = y.shape
    y_c = y.contiguous()
    pos_c = pos_slice.contiguous()
    stride_yb, stride_yt, stride_ym = y_c.stride()
    stride_pos_t, stride_pos_m = pos_c.stride()
    grid = (B, T, triton.cdiv(M, 64))
    add_scaled_pos_emb_kernel[grid](
        y_c, pos_c, float(scale),
        B, T, M,
        stride_yb, stride_yt, stride_ym,
        stride_pos_t, stride_pos_m,
        num_warps=4, num_stages=2
    )
    return y_c


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        x = args[0]
        conv1_w, conv1_b = args[1], args[2]
        conv2_w, conv2_b = args[3], args[4]
        conv3_w, conv3_b = args[5], args[6]
        conv_out_w = args[7]  # (d_model=1024, K=3840)
        pos_emb = args[8]     # (max_source_positions=1500, d_model=1024)
        embed_scale = args[9]  # float

        # Stage 1: Conv2d (1 -> 384) + GELU
        x = F.conv2d(x, conv1_w, conv1_b, stride=2, padding=1)
        x = triton_gelu(x)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2_w, conv2_b, stride=2, padding=1)
        x = triton_gelu(x)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv3_w, conv3_b, stride=2, padding=1)
        x = triton_gelu(x)

        # Reshape: (B, C=384, F=10, T=T_after_conv) -> (B, T, 384*10)
        b, c, f, t = x.size()
        M = c * f  # 3840
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, M)

        # Linear projection (no bias) via Triton using conv_out_weight (1024, 3840)
        # Important: conv_out_weight is (d_model=1024, K=3840). We need Y of shape (B, T, 1024).
        # However, the original run(...) linear uses conv_out_weight (d_model, conv_out_dim=3840)
        # But the provided get_inputs returns conv_out_weight shape (d_model=1024, K=3840).
        # We will use this weight directly. If your actual conv_out_dim is not 3840, adjust inputs.
        y = triton_linear_no_bias(x, conv_out_w)

        # Scale by embed_scale and add positional embedding
        # pos_emb shape: (1500, 1024), slice first T rows
        pos_slice = pos_emb[:t, :].contiguous()
        scale = float(embed_scale)
        y = triton_add_scaled_pos_emb(y, pos_slice, scale)

        return y


def run(*args):
    return ModelNew()(*args)
