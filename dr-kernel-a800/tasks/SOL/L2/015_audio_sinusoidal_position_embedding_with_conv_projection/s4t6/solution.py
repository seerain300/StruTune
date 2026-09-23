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
    # gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
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
    BLOCK_M: tl.constexpr,  # tile over output channels (M)
    BLOCK_K: tl.constexpr   # tile over reduction (K)
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

    # Reduce over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_chunk = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets]
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_chunk = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # acc[m] += sum_k w[m, k] * x[k]
        acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)  # sum over K tile

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale and add positional embedding
# Input: Y_scaled (B, T_after_conv, d_model) already scaled
# PosEmbed: (pos_len, d_model) where pos_len = T_after_conv
# Output: Y_out (B, T_after_conv, d_model) = Y_scaled + PosEmbed
@triton.jit
def add_pos_emb_kernel(
    Y_scaled, PosEmb, Y_out,
    B, T, d_model,
    stride_ysb, stride_yst, stride_ysd,
    stride_pep, stride_ped,  # pos embedding is (T, d_model)
    stride_yob, stride_yot, stride_yod,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t = pid_t
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_model

    y_scaled_ptrs = Y_scaled + b * stride_ysb + t * stride_yst + d_offsets * stride_ysd
    y_scaled = tl.load(y_scaled_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # pos index equals t since T == time_after_conv in our pipeline
    pos = t
    pos_emb_ptrs = PosEmb + pos * stride_pep + d_offsets * stride_ped
    pos_emb = tl.load(pos_emb_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    out = y_scaled + pos_emb

    y_out_ptrs = Y_out + b * stride_yob + t * stride_yot + d_offsets * stride_yod
    tl.store(y_out_ptrs, out, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.batch_size = axes_and_scalars["batch_size"]
        self.time_dim = axes_and_scalars["time_dim"]
        self.device = device
        self.dtype = dtype
        # Precompute embed_scale
        self.embed_scale = math.sqrt(1024)  # d_model = 1024

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor,
        conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor,
        conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor,
        conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,  # (1024, 3840)
        positional_embedding: torch.Tensor,  # (1500, 1024) in bfloat16
    ):
        # Stage 1: Conv2d (1 -> 384) + GELU
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        y1 = torch.empty_like(x1, dtype=torch.bfloat16, device=self.device)
        N1 = x1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x1, y1, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x2 = F.conv2d(y1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        y2 = torch.empty_like(x2, dtype=torch.bfloat16, device=self.device)
        N2 = x2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x2, y2, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x3 = F.conv2d(y2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        y3 = torch.empty_like(x3, dtype=torch.bfloat16, device=self.device)
        N3 = x3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x3, y3, N3, BLOCK=1024)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = y3.size()  # c=384, f=10, t = T//8
        x_proj = y3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias), Triton GEMM-like kernel
        B = self.batch_size
        T_after = t  # time_after_conv
        K = c * f     # 3840
        M = 1024
        X = x_proj.contiguous()  # (B, T_after, K), bfloat16
        W = conv_out_weight.contiguous()  # (M, K), bfloat16
        Y_lin = torch.empty((B, T_after, M), dtype=torch.bfloat16, device=self.device)

        # Strides
        stride_xb, stride_xt, stride_xk = X.stride()
        stride_wm, stride_wk = W.stride()
        stride_yb, stride_yt, stride_ym = Y_lin.stride()

        # Grid: (B, T_after, ceil(M/BLOCK_M))
        BLOCK_M = 64
        BLOCK_K = 128
        grid = (B, T_after, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            X, W, Y_lin,
            B, T_after, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Scale embeddings: Y_lin *= embed_scale
        Y_scaled = Y_lin * self.embed_scale

        # Add positional embeddings: pos index equals t (since T == time_after_conv)
        # pos_emb is (1500, 1024), we only need first T_after rows
        pos_emb = positional_embedding[:T_after, :].to(torch.bfloat16).contiguous()
        Y_out = torch.empty((B, T_after, 1024), dtype=torch.bfloat16, device=self.device)

        # Strides for Y_scaled and pos_emb
        y_scaled = Y_scaled
        stride_ysb, stride_yst, stride_ysd = y_scaled.stride()
        stride_pep, stride_ped = pos_emb.stride()
        stride_yob, stride_yot, stride_yod = Y_out.stride()

        BLOCK_D = 256
        grid_pos = (B, T_after, triton.cdiv(1024, BLOCK_D))
        add_pos_emb_kernel[grid_pos](
            y_scaled, pos_emb, Y_out,
            B, T_after, 1024,
            stride_ysb, stride_yst, stride_ysd,
            stride_pep, stride_ped,
            stride_yob, stride_yot, stride_yod,
            BLOCK_D=BLOCK_D,
        )

        return Y_out


def run(*args):
    return ModelNew()(*args)
