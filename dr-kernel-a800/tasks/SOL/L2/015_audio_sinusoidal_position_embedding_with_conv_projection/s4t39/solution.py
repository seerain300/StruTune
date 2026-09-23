import math
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise over a flattened tensor
# Inputs: X (flattened), Outputs: Y (flattened)
# N: total number of elements
@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    x3 = x * x * x
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, y, mask=mask)


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K) — flattened to (B*T, K)
# W: (M, K) — conv_out_weight (1024, 3840)
# Y: (B, T, M) — flattened to (B*T, M)
# We tile over M (output channels) and reduce over K in chunks.
@triton.jit
def linear_no_bias_kernel_flat(
    X, W, Y,
    B, T, K, M,
    stride_xbt, stride_xk,    # X strides: (B*T, K)
    stride_wm, stride_wk,     # W strides: (M, K)
    stride_ybt, stride_ym,    # Y strides: (B*T, M)
    BLOCK_M: tl.constexpr,    # tile size for M
    BLOCK_K: tl.constexpr     # chunk size for reduction over K
):
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # time index
    pid_m = tl.program_id(2)  # tile over output channels

    bt = pid_b * T + pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[bt, k_offsets] -> shape (BLOCK_K,)
        x_ptrs = X + bt * stride_xbt + k_offsets * stride_xk
        x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_vals = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # Accumulate: acc[m] += sum_k x_vals[k] * w_vals[m, k]
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Store acc to Y[bt, m_offsets]
    y_ptrs = Y + bt * stride_ybt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale by scalar and add positional embedding (elementwise)
# Y_in: (B, T, M) bfloat16 input (already scaled and prepared)
# POS: (M, L) L = T (time_after_conv), we load POS[:, t] per batch
# Y_out: (B, T, M) bfloat16
# We assume M=1024 is known and loop over m and t; for performance, we could vectorize over M tiles, but this is simpler and robust.
@triton.jit
def scale_add_pos_emb_kernel(
    Y_in, POS, Y_out,
    B, T, M, L,
    stride_yb, stride_yt, stride_ym,
    stride_posm, stride_posl,
    embed_scale,
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

    # Load Y_in[b, t, m_offsets]
    y_ptrs = Y_in + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    y_vals = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Load POS[m_offsets, t] -> vector of length BLOCK_M
    pos_ptrs = POS + m_offsets * stride_posm + t * stride_posl
    pos_vals = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Compute: y = y_vals * embed_scale + pos_vals
    y_new = y_vals * embed_scale + pos_vals

    # Store back (cast to original dtype if needed)
    out_ptrs = Y_out + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(out_ptrs, y_new, mask=m_mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Stage 1: Conv2d (1 -> 384) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = self._gelu_triton(x)  # Triton GELU

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = self._gelu_triton(x)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = self._gelu_triton(x)

        # Reshape: (B, C, F, T) -> (B, T, C*F)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T_after_conv, 3840)

        # Linear projection (no bias) in Triton: (B, T, 3840) @ (1024, 3840)^T -> (B, T, 1024)
        M = conv_out_weight.shape[0]  # 1024
        K = x.shape[2]                # 3840
        B_T = B * T

        # Allocate output (float32 for accumulation), then cast to bfloat16 before adding positional embedding
        Y_fp32 = torch.empty((B, T, M), dtype=torch.float32, device=x.device)

        # Prepare inputs: X as (B*T, K), W as (M, K)
        X_flat = x.to(torch.float32).contiguous().view(B_T, K)
        W = conv_out_weight.contiguous()  # (M, K)

        BLOCK_M = 128
        BLOCK_K = 128
        grid = (B, T, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel_flat[grid](
            X_flat, W, Y_fp32,
            B, T, K, M,
            X_flat.stride(0), X_flat.stride(1),
            W.stride(0), W.stride(1),
            Y_fp32.stride(0), Y_fp32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
        )

        # Scale by embed_scale (float32 for safety)
        Y_scaled_fp32 = Y_fp32 * embed_scale  # embed_scale is a Python float, multiply fp32

        # Convert to bfloat16 for final add with positional embedding
        Y_bf16 = Y_scaled_fp32.to(torch.bfloat16)  # (B, T, M)

        # Add positional embedding: shape (M, L), L = T
        # We need to add POS[:T, :] to each batch row. Launch Triton elementwise kernel.
        M = Y_bf16.shape[2]
        L = T
        POS = positional_embedding.to(torch.bfloat16)  # (M, 1500)
        Y_out = torch.empty_like(Y_bf16)

        grid2 = (B, T, triton.cdiv(M, BLOCK_M))
        scale_add_pos_emb_kernel[grid2](
            Y_bf16, POS, Y_out,
            B, T, M, L,
            Y_bf16.stride(0), Y_bf16.stride(1), Y_bf16.stride(2),
            POS.stride(0), POS.stride(1),
            float(embed_scale),  # embed_scale as float
            BLOCK_M=BLOCK_M
        )

        return Y_out

    def _gelu_triton(self, x: torch.Tensor) -> torch.Tensor:
        # Elementwise GELU via Triton
        y = torch.empty_like(x)
        N = x.numel()
        # Ensure contiguous
        x_c = x.contiguous()
        y_c = y.contiguous()
        # Launch Triton kernel
        gelu_tanh_kernel[(triton.cdiv(N, 1024),)](x_c, y_c, N, BLOCK=1024)
        return y_c


def run(*args):
    return ModelNew()(*args)
