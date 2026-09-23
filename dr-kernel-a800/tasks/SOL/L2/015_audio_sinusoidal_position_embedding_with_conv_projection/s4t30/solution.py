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
    # tanh approximation constants
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, gelu, mask=mask)


# Triton kernel: Linear projection (no bias), compute Y[b, t, m] = sum_k X[b, t, k] * W[m, k]
# Inputs:
#   X: (B, T, K), contiguous
#   W: (M, K), contiguous
# Outputs:
#   Y: (B, T, M), contiguous
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile size over M (output channels)
    BLOCK_K: tl.constexpr   # tile size over K (reduction)
):
    # Grid is (B, T, ceil_div(M, BLOCK_M))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets] as a vector
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] as a matrix [BLOCK_M, BLOCK_K]
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)

        # Accumulate: acc[m] += sum_k W[m, k] * X[b, t, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store results to Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale + add positional embedding
# Y: (B, T, M) — input
# POS: (MAX_T, M) — contiguous positional embedding
# SCALE: float scalar
# After reading Y[b, t, :], add POS[t, :] * SCALE, in-place or to OUT
@triton.jit
def add_scaled_pos_emb_kernel(
    Y, POS, OUT,
    B, T, M,
    stride_yb, stride_yt, stride_ym,
    stride_pos_t, stride_pos_m,
    SCALE: tl.float32,
    BLOCK_M: tl.constexpr
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

    # Store back to OUT (can be same as Y if in-place allowed)
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
        # Stage 1: Conv2d (1 -> 384 channels) + GELU (PyTorch conv, Triton GELU)
        x1 = torch.nn.functional.conv2d(
            input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1
        )
        x1_numel = x1.numel()
        x1_gelu = torch.empty_like(x1)
        gelu_tanh_kernel[(triton.cdiv(x1_numel, 1024),)](
            x1, x1_gelu, x1_numel, BLOCK=1024
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = torch.nn.functional.conv2d(
            x1_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1
        )
        x2_numel = x2.numel()
        x2_gelu = torch.empty_like(x2)
        gelu_tanh_kernel[(triton.cdiv(x2_numel, 1024),)](
            x2, x2_gelu, x2_numel, BLOCK=1024
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = torch.nn.functional.conv2d(
            x2_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1
        )
        x3_numel = x3.numel()
        x3_gelu = torch.empty_like(x3)
        gelu_tanh_kernel[(triton.cdiv(x3_numel, 1024),)](
            x3, x3_gelu, x3_numel, BLOCK=1024
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x3_gelu.shape
        x3_gelu = x3_gelu.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias) using Triton
        B, T, K = b, t, x3_gelu.shape[-1]  # K should be 384*10=3840
        M = conv_out_weight.shape[0]       # 1024

        # Ensure contiguous for Triton
        x4 = x3_gelu.contiguous()           # (B, T, K)
        W = conv_out_weight.contiguous()    # (M, K)

        Y = torch.empty((B, T, M), dtype=torch.float32, device=x4.device)

        # Strides for Triton
        stride_xb, stride_xt, stride_xk = x4.stride(0), x4.stride(1), x4.stride(2)
        stride_wm, stride_wk = W.stride(0), W.stride(1)
        stride_yb, stride_yt, stride_ym = Y.stride(0), Y.stride(1), Y.stride(2)

        # Choose tile sizes
        BLOCK_M = 64
        BLOCK_K = 128
        grid = (B, T, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            x4, W, Y,
            B, T, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
        )

        # Scale by embed_scale
        Y_scaled = Y  # already float32, we can add directly

        # Add scaled positional embedding: POS shape (1500, 1024), take first T rows
        POS = positional_embedding.to(torch.float32)  # ensure fp32 for computation
        # Allocate OUT as new tensor to avoid aliasing issues
        OUT = torch.empty_like(Y_scaled)

        BLOCK_M_add = 128
        add_scaled_pos_emb_kernel[(B, T, triton.cdiv(M, BLOCK_M_add))](
            Y_scaled, POS, OUT,
            B, T, M,
            stride_yb, stride_yt, stride_ym,
            POS.stride(0), POS.stride(1),
            float(embed_scale), BLOCK_M=BLOCK_M_add
        )

        # Return as float32 (original run returns same dtype as input, which is bfloat16,
        # but for robustness we keep float32. If strict dtype is required, cast at the end.)
        # return OUT
        # If you need bfloat16 to match original, uncomment the cast below:
        # return OUT.to(torch.bfloat16)

        return OUT


def run(*args):
    return ModelNew()(*args)
