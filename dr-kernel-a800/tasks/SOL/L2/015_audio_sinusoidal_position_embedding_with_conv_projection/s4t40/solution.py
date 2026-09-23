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
    # tanh approximation of GELU
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, gelu, mask=mask)


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K) where T is time_after_conv, K=3840
# W: (M, K) where M=1024
# Y: (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile over output channels
    BLOCK_K: tl.constexpr   # tile over reduction dim
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

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
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_block = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc[m] += sum_k (x_vec[k] * w_block[m, k])
        acc += tl.sum(w_block * x_vec[None, :], axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise add scaled pos embedding
# Y: (B, T, M), Pos: (S, M), scale: scalar
# For each (b, t, m): Y[b,t,m] += scale * Pos[t, m]
@triton.jit
def add_scaled_pos_emb_kernel(Y, Pos, SCALE, B, T, M, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    y_ptrs = Y + pid_b * (T * M) + pid_t * M + m_offsets
    pos_ptrs = Pos + pid_t * M + m_offsets

    y = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    y = y + SCALE * pos
    tl.store(y_ptrs, y, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        x = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # Triton GELU after conv1
        y1 = torch.empty_like(x)
        N1 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x, y1, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = torch.nn.functional.conv2d(y1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        y2 = torch.empty_like(x)
        N2 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x, y2, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = torch.nn.functional.conv2d(y2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        y3 = torch.empty_like(x)
        N3 = x.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x, y3, N3, BLOCK=1024)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = y3.size()
        y3 = y3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias) — use Triton
        B = y3.shape[0]
        T_after = y3.shape[1]
        K = y3.shape[2]  # 3840
        M = conv_out_weight.shape[0]  # 1024

        # Ensure contiguous for simple stride computation
        X = y3.contiguous()
        W = conv_out_weight.contiguous()
        Y_lin = torch.empty((B, T_after, M), dtype=torch.bfloat16, device=y3.device)

        stride_xb, stride_xt, stride_xk = X.stride()
        stride_wm, stride_wk = W.stride()
        stride_yb, stride_yt, stride_ym = Y_lin.stride()

        # Launch Triton linear_no_bias_kernel
        grid = (B, T_after, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid](
            X, W, Y_lin,
            B, T_after, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=128,
            BLOCK_K=256,
        )

        # Scale embeddings
        # Cast to float32 for math, Triton will produce bf16 output
        # We can scale in-place on the bf16 output: Triton add_scaled_pos_emb_kernel handles scaling
        # But to ensure type safety, we compute scale as float32 scalar and pass it to kernel

        # Add scaled positional embeddings (only using first T_after rows)
        pos_slice = positional_embedding[:T_after, :].contiguous()
        # Triton kernel expects bf16 or fp32; we pass bf16 and scale as float32
        # Ensure Y_lin is the correct dtype; conv_out_weight was float32; we cast pos_slice to bf16
        pos_slice = pos_slice.to(torch.bfloat16)

        # Scale as float32
        SCALE = float(embed_scale)

        add_scaled_pos_emb_kernel[(B, T_after, triton.cdiv(M, 128))](Y_lin, pos_slice, SCALE, B, T_after, M, BLOCK_M=128)

        return Y_lin


def run(*args):
    return ModelNew()(*args)
