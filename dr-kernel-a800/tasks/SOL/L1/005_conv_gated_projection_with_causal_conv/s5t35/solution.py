import torch
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_xm, stride_xn,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over IN_H in tiles
    for k0 in range(0, IN_H, 128):
        offs_k = k0 + tl.arange(0, 128)
        k_mask = offs_k < IN_H
        # Load X block: [BLOCK_M, 128]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, 128]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T, i.e., sum over k
        # x: [BM, BK], w: [BN, BK] -> acc += sum_k x[:, k] * w[:, k]^T
        # Practical: compute per-k and accumulate
        for kk in range(128):
            if kk < IN_H:
                x_col = x[:, kk]  # [BM]
                w_col = w[:, kk]  # [BN]
                acc += x_col[:, None] * w_col[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    IN_ptr, OUT1_ptr, OUT2_ptr, OUT3_ptr,
    M, C,  # M = B * S, C = H
    stride_im, stride_in,  # IN is (M, 3*C)
    stride_ob1m, stride_ob1n,  # OUT1 (B, C, S) but we will flatten (M, C)
    stride_ob2m, stride_ob2n,  # OUT2 similarly
    stride_ob3m, stride_ob3n,  # OUT3 similarly
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C], OUTi: [M, C]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    # Load IN block and write to three outputs corresponding to channels 0..C-1, C..2C-1, 2C..3C-1
    # We will compute for each n in [0..C-1]:
    for n0 in range(0, C, BLOCK_N):
        nn = n0 + offs_n
        nn_mask = nn < C
        # Combine masks
        mm_mask = offs_m < M

        # Load IN for each slice
        in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + nn[None, :] * stride_in)  # corresponds to first C
        in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (C + nn[None, :]) * stride_in)  # second C
        in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (2*C + nn[None, :]) * stride_in)  # third C

        x1 = tl.load(in_ptrs1, mask=mm_mask[:, None] & nn_mask[None, :], other=0.0)
        x2 = tl.load(in_ptrs2, mask=mm_mask[:, None] & nn_mask[None, :], other=0.0)
        x3 = tl.load(in_ptrs3, mask=mm_mask[:, None] & nn_mask[None, :], other=0.0)

        # Store to OUT1 (first C), OUT2 (middle C), OUT3 (last C)
        out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_ob1m + nn[None, :] * stride_ob1n)
        out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_ob2m + nn[None, :] * stride_ob2n)
        out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_ob3m + nn[None, :] * stride_ob3n)

        tl.store(out1_ptrs, x1, mask=mm_mask[:, None] & nn_mask[None, :])
        tl.store(out2_ptrs, x2, mask=mm_mask[:, None] & nn_mask[None, :])
        tl.store(out3_ptrs, x3, mask=mm_mask[:, None] & nn_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # A: [M, N], B: [M, N], OUT: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = a * b

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, C_in, L, PAD, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] where M = B*S, C_in = H, L = S
    # W_ptr: [C_in, 1, K] (groups=C_in), but we pass stride_wK for K and ignore the 1
    # BIAS_ptr: [C_in], OUT_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    # Accumulator for [BLOCK_M, BLOCK_C]
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # We implement explicit left-pad with PAD=3 and no right-pad.
    # For each output time t (we iterate tiles along L), accumulate over k in {0..K-1}.
    for t0 in range(0, L, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        ti_mask = t < L

        for k in range(0, K):
            t_in = t + PAD - k  # left-pad of 3, causal => k in {0,1,2,3}
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + tl.arange(0, BLOCK_T))[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + tl.arange(0, BLOCK_T))[None, :] < L
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over IN_H in tiles
    for k0 in range(0, IN_H, 128):
        offs_k = k0 + tl.arange(0, 128)
        k_mask = offs_k < IN_H

        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BM, BK]

        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BN, BK]

        # acc += x @ w^T
        for kk in range(128):
            if kk < IN_H:
                x_col = x[:, kk]  # [BM]
                w_col = w[:, kk]  # [BN]
                acc += x_col[:, None] * w_col[None, :]

    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                 conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                 out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj_weight = in_proj_weight  # (3*hidden_size, hidden_size)
        self.in_proj_bias = in_proj_bias      # (3*hidden_size,)
        self.conv_weight = conv_weight        # (hidden_size, 1, 4), groups=hidden_size
        self.conv_bias = conv_bias            # (hidden_size,)
        self.out_proj_weight = out_proj_weight  # (hidden_size, hidden_size)
        self.out_proj_bias = out_proj_bias      # (hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shapes
        B, S, H = x.shape
        M = B * S
        assert H == self.hidden_size

        # 1) in_proj: y = F.linear(x, in_proj_weight, in_proj_bias)
        # x_flat: (M, H), weight: (M_out, H) where M_out = 3*H
        x_flat = x.reshape(M, H).contiguous()
        M_out = self.in_proj_weight.shape[0]  # 3*H
        y_flat = torch.empty((M, M_out), dtype=x_flat.dtype, device=x_flat.device)

        # Launch GEMM + bias
        BLOCK_M = 128
        BLOCK_N = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        linear_kernel[grid](
            x_flat, self.in_proj_weight, self.in_proj_bias, y_flat,
            M, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            self.in_proj_weight.stride(0), self.in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 2) chunk along dim=1 to get B, C, x_proj
        # y_flat: (M, 3*H), chunk into three: (M, H) each
        B_ = torch.empty((M, H), dtype=y_flat.dtype, device=y_flat.device)
        C_ = torch.empty((M, H), dtype=y_flat.dtype, device=y_flat.device)
        x_proj = torch.empty((M, H), dtype=y_flat.dtype, device=y_flat.device)

        grid_chunk = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        chunk_dim1_3_kernel[grid_chunk](
            y_flat, B_, C_, x_proj,
            M, H,
            y_flat.stride(0), y_flat.stride(1),
            B_.stride(0), B_.stride(1),
            C_.stride(0), C_.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Reshape back to (B, S, H)
        B_ = B_.reshape(B, S, H)
        C_ = C_.reshape(B, S, H)
        x_proj = x_proj.reshape(B, S, H)

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x_flat.dtype, device=x_flat.device)
        grid_mul = (triton.cdiv(B*S, BLOCK_M), triton.cdiv(H, BLOCK_N))
        mul_elementwise_kernel[grid_mul](
            B_, x_proj, Bx,
            B*S, H,
            B_.stride(0), B_.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 4) Grouped causal 1D convolution with kernel_size=4, pad=(3,0), groups=H
        # Input X: (M, H, S), weight: (H, 1, 4), bias: (H,), output: (M, H, S)
        X_padded = Bx  # already left-pad equivalent via kernel by using PAD=3
        L = S
        C_in = H
        K = 4
        PAD = 3

        conv_out_flat = torch.empty((M, C_in), dtype=x_flat.dtype, device=x_flat.device)

        grid_conv = (triton.cdiv(M, 128), triton.cdiv(C_in, 64))
        grouped_causal_conv1d_kernel[grid_conv](
            X_padded, self.conv_weight, self.conv_bias, conv_out_flat,
            M, C_in, L, PAD, K,
            X_padded.stride(0), X_padded.stride(1), X_padded.stride(2),
            self.conv_weight.stride(0), self.conv_weight.stride(2),  # stride_wK corresponds to kernel dim
            conv_out_flat.stride(0), conv_out_flat.stride(1), conv_out_flat.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128,
        )

        conv_out = conv_out_flat.reshape(B, H, S)

        # 5) Output gating: y = C * conv_out
        y = C_ * conv_out  # (B, H, S)

        # 6) Final output projection: y -> out
        # y_flat: (B*S, H)
        y_flat = y.reshape(M, H).contiguous()
        out = torch.empty((M, H), dtype=x_flat.dtype, device=x_flat.device)

        grid_final = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        final_proj_kernel[grid_final](
            y_flat, self.out_proj_weight, self.out_proj_bias, out,
            M, H, H,
            y_flat.stride(0), y_flat.stride(1),
            self.out_proj_weight.stride(0), self.out_proj_weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        return out.reshape(B, S, H)


# Helper functions to generate inputs with given batch_size and seq_len
def build_in_proj(hidden_size: int, device='cuda'):
    # in_proj_weight: (3*hidden_size, hidden_size), in_proj_bias: (3*hidden_size,)
    in_proj_weight = torch.randn(3 * hidden_size, hidden_size, device=device, dtype=torch.float32)
    in_proj_bias = torch.randn(3 * hidden_size, device=device, dtype=torch.float32)
    return in_proj_weight, in_proj_bias

def build_conv(hidden_size: int, device='cuda'):
    # conv_weight: (hidden_size, 1, 4), conv_bias: (hidden_size,)
    conv_weight = torch.randn(hidden_size, 1, 4, device=device, dtype=torch.float32)
    conv_bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
    return conv_weight, conv_bias

def build_out_proj(hidden_size: int, device='cuda'):
    # out_proj_weight: (hidden_size, hidden_size), out_proj_bias: (hidden_size,)
    out_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
    out_proj_bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
    return out_proj_weight, out_proj_bias


# Example usage for evaluation:
# model = ModelNew(hidden_size=H, in_proj_weight=..., in_proj_bias=..., conv_weight=..., conv_bias=..., out_proj_weight=..., out_proj_bias=...)
# x = torch.randn(B, S, H, device='cuda', dtype=torch.float32)
# y = model(x)


def run(*args):
    return ModelNew()(*args)
