import math
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation) elementwise over N elements
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
# X: (B, T, K) — bfloat16 input, we load as f32 for accumulation
# W: (M, K) — conv_out_weight (1024, 3840)
# Y: (B, T, M) — bfloat16 output
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,   # tile size for M (output channels)
    BLOCK_K: tl.constexpr    # chunk size for reduction over K
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # time index
    pid_m = tl.program_id(2)  # tile over output channels (M)

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

        # Load X[b, t, k_offsets] as float32
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[m_offsets, k_offsets] as float32
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        load_mask = m_mask[:, None] & k_mask[None, :]
        w_mat = tl.load(w_ptrs, mask=load_mask, other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # Accumulate: acc[m] += sum_k w[m,k] * x[t,k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store to Y as bfloat16
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Final fusion — scale and add positional embedding:
# Final = Y_scaled + pos_emb[:T_after_conv, :]
# Y_scaled = Y * scale
# pos_emb is (T_after_conv, M), bfloat16
@triton.jit
def scale_add_pos_emb_kernel(
    Y, pos_emb, Final,
    B, T, M,
    stride_yb, stride_yt, stride_ym,
    stride_pesb, stride_pest, stride_pems,  # pos_emb is (T, M)
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # time
    pid_m = tl.program_id(2)  # channel tile

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Load Y[b, t, m_offsets] as float32
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    y_val = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Scale
    y_scaled = y_val * scale

    # Load pos_emb[t, m_offsets] as float32
    pe_ptrs = pos_emb + t * stride_pest + m_offsets * stride_pems
    pe_val = tl.load(pe_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Add
    out_val = y_scaled + pe_val

    # Store as bfloat16 to Final (Final has same layout as Y)
    out_ptrs = Final + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(out_ptrs, out_val, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, *args):
        # Extract inputs
        input_features = args[0]  # (B, 1, 80, T), bfloat16
        conv2d1_weight, conv2d1_bias = args[1], args[2]
        conv2d2_weight, conv2d2_bias = args[3], args[4]
        conv2d3_weight, conv2d3_bias = args[5], args[6]
        conv_out_weight = args[7]  # (1024, 3840)
        positional_embedding = args[8]  # (1500, 1024), bfloat16
        embed_scale = float(args[9])  # python float

        B, _, H1, W1 = input_features.shape
        T = W1

        # Stage 1: Conv2d (1 -> 384) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = x.contiguous()
        N1 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x, x_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = x.contiguous()
        N2 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x, x_gelu, N2, BLOCK=1024)
        x = x_gelu

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = x.contiguous()
        N3 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x, x_gelu, N3, BLOCK=1024)
        x = x_gelu

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10) i.e., (B, T_after_conv, 3840)
        b, c, f, t = x.size()  # c=384, f=10, t=T//8
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T//8, 3840)

        # Linear projection to d_model=1024 (no bias) via Triton
        B2, T_after_conv, K = x.shape  # K = 3840
        M = 1024  # d_model

        # Prepare output (bfloat16)
        y = torch.empty((B2, T_after_conv, M), dtype=torch.bfloat16, device=self.device)

        # Strides
        x_strides = x.stride()  # (B, T, K)
        w = conv_out_weight  # (M, K)
        w_strides = w.stride()  # (M, K)
        y_strides = y.stride()  # (B, T, M)

        # Launch Triton kernel
        grid = (B2, T_after_conv, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid](
            x, w, y,
            B2, T_after_conv, K, M,
            x_strides[0], x_strides[1], x_strides[2],
            w_strides[0], w_strides[1],
            y_strides[0], y_strides[1], y_strides[2],
            BLOCK_M=128, BLOCK_K=128
        )

        # Prepare final output with scaled add of positional embedding
        final = torch.empty_like(y)

        # Slice positional embedding to first T_after_conv rows
        pos_sliced = positional_embedding[:T_after_conv, :].to(torch.bfloat16).to(self.device)

        # Strides for addition kernel
        y_strides_out = y.stride()  # same layout as y
        pos_strides = pos_sliced.stride()

        # Launch Triton kernel: final = y * scale + pos_emb[:T_after_conv, :]
        scale_add_pos_emb_kernel[(B2, T_after_conv, triton.cdiv(M, 128))](
            y, pos_sliced, final,
            B2, T_after_conv, M,
            y_strides_out[0], y_strides_out[1], y_strides_out[2],
            pos_strides[0], pos_strides[1], pos_strides[2],
            scale=embed_scale,
            BLOCK_M=128
        )

        return final


def run(*args):
    return ModelNew()(*args)
