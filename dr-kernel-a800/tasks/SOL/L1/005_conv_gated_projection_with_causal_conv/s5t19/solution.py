import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_out,
    stride_xm, stride_xn,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute OUT[M, M_out] = X[M, H] @ W[M_out, H]^T + BIAS[M_out]
    M = B * S, H = input hidden size, M_out = 3 * H.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # over M_out
    offs_k = tl.arange(0, BLOCK_K)  # over H

    m_mask = offs_m < M

    # Accumulator for each output n (M_out) across M blocks
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over H in chunks
    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < H

        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K] (W is [M_out, H])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=(offs_n < M_out)[:, None] & k_mask[None, :], other=0.0)

        # acc += X @ W^T -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Add bias: [BLOCK_N]
    bias = tl.load(BIAS_ptr + offs_n, mask=(offs_n < M_out), other=0.0)
    acc = acc + bias[None, :]

    # Store OUT
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = m_mask[:, None] & (offs_n < M_out)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def chunk_dim1_3_kernel(
    Y_ptr, B_ptr, C_ptr, XPRJ_ptr,
    M, H,
    stride_yM, stride_yN,
    stride_bM, stride_bN,
    stride_cM, stride_cN,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Split Y[M, 3H] into three chunks along dim=1: B[M, H], C[M, H], XPRJ[M, H].
    """
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    # Load Y block and store into B, C, XPRJ
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + offs_n[None, :] * stride_yN)
    y_vals = tl.load(y_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_N]

    # Store B
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bM + offs_n[None, :] * stride_bN)
    tl.store(b_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :])

    # Store C (next H chunk of Y)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cM + offs_n[None, :] * stride_cN)
    tl.store(c_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :])

    # Store XPRJ (last H chunk of Y)
    xprj_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    tl.store(xprj_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_bn,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    OUT[M, H] = B[M, H] * XPRJ[M, H]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    x_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)

    b_vals = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out_vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """
    X: [M, C_in, L] where M = B*S, C_in = H, L = seq_len
    W: [C_in, K] (conv_weight per output channel)
    Out: [M, C_in, L]
    groups = C_in (depthwise), no padding (pad=0).
    """
    # grid = (ceil(M/BLOCK_M), ceil(C_in/BLOCK_C))
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Tile over output positions t
    for t0 in range(0, L, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        t_mask = offs_t < L

        # For each kernel position k
        for k in range(0, K):
            t_in = offs_t - k  # causal
            valid = (t_in >= 0) & (t_in < L) & t_mask

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
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + offs_t)[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + offs_t)[None, :] < L
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    OUT[M, OUT_H] = IN[M, IN_H] @ W[OUT_H, IN_H]^T + BIAS[OUT_H]
    M = B * S, IN_H = H, OUT_H = H.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # over OUT_H
    offs_k = tl.arange(0, BLOCK_K)  # over IN_H

    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K], W is [OUT_H, IN_H]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=(offs_n < OUT_H)[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(x, tl.trans(w))

    # Add bias: [BLOCK_N]
    bias = tl.load(BIAS_ptr + offs_n, mask=(offs_n < OUT_H), other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_n < OUT_H))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        x: (B, S, H)
        in_proj_weight: (M_out, H), M_out = 3 * H
        in_proj_bias: (M_out,)
        conv_weight: (H, K), K = 4, groups = H
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        Returns: (B, S, H)
        """
        assert x.dim() == 3, "x must be (B, S, H)"
        B, S, H = x.shape
        M = B * S

        # Ensure dtype float32
        device = x.device
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.float()
            in_proj_weight = in_proj_weight.float()
            in_proj_bias = in_proj_bias.float()
            conv_weight = conv_weight.float()
            conv_bias = conv_bias.float()
            out_proj_weight = out_proj_weight.float()
            out_proj_bias = out_proj_bias.float()

        # 1) in_proj linear: Y_flat = X_flat @ in_proj_weight^T + in_proj_bias
        M_out = 3 * H
        X_flat = x.reshape(M, H).contiguous()
        Y_flat = torch.empty((M, M_out), device=device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))

        in_proj_linear_kernel[grid](
            X_flat, in_proj_weight, in_proj_bias, Y_flat,
            M, H, M_out,
            X_flat.stride(0), X_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            Y_flat.stride(0), Y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Split Y_flat into B, C, x_proj (each shape (M, H))
        Y = Y_flat.view(M, 3 * H)
        B_ = torch.empty((M, H), device=device, dtype=torch.float32)
        C_ = torch.empty((M, H), device=device, dtype=torch.float32)
        XPRJ_ = torch.empty((M, H), device=device, dtype=torch.float32)

        # Grid for chunk along H
        BLOCK_M_chunk = 128
        BLOCK_N_chunk = 64
        grid_chunk = (triton.cdiv(M, BLOCK_M_chunk), triton.cdiv(H, BLOCK_N_chunk))

        chunk_dim1_3_kernel[grid_chunk](
            Y, B_, C_, XPRJ_,
            M, H,
            Y.stride(0), Y.stride(1),
            B_.stride(0), B_.stride(1),
            C_.stride(0), C_.stride(1),
            XPRJ_.stride(0), XPRJ_.stride(1),
            BLOCK_M=BLOCK_M_chunk, BLOCK_N=BLOCK_N_chunk,
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((M, H), device=device, dtype=torch.float32)

        grid_mul = (triton.cdiv(M, BLOCK_M_chunk), triton.cdiv(H, BLOCK_N_chunk))
        mul_elementwise_kernel[grid_mul](
            B_, XPRJ_, Bx,
            M, H,
            B_.stride(0), B_.stride(1),
            XPRJ_.stride(0), XPRJ_.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=BLOCK_M_chunk, BLOCK_N=BLOCK_N_chunk,
        )

        # 4) Grouped causal conv1d: groups=H, kernel_size=4, no padding (pad=0)
        # Bx: [M, H, S] (we need to feed as [M, C_in, L])
        Bx_reshaped = Bx.view(M, H).contiguous()  # (M, H)
        conv_out = torch.empty((M, H), device=device, dtype=torch.float32)

        K = conv_weight.shape[1]
        grid_conv = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_reshaped, conv_weight, conv_bias, conv_out,
            M, H, S, K,
            Bx_reshaped.stride(0), Bx_reshaped.stride(1), S,
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), S,
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128,
        )

        # Reshape conv_out to (B, H, S)
        conv_out = conv_out.view(B, S, H)

        # 5) Output gating: y = C * conv_out
        y = C_.view(B, S, H) * conv_out

        # 6) Final projection: y -> (B, S, H)
        y_flat = y.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), device=device, dtype=torch.float32)

        grid_fin = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        final_proj_kernel[grid_fin](
            y_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            y_flat.stride(0), y_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        out = out_flat.view(B, S, H)
        # Cast back to original dtype if needed
        if dtype != torch.float32:
            out = out.to(dtype)
        return out


def run(*args):
    return ModelNew()(*args)
