import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_out,
    stride_xm, stride_xn,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    OUT[M, M_out] = X[M, H] @ W[M_out, H]^T + BIAS[M_out]
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, M_out), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H

        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block as (BLOCK_K, M_out)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wm + tl.arange(0, M_out)[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (tl.arange(0, M_out)[None, :] < M_out), other=0.0)

        acc += tl.dot(x, tl.trans(w))  # [BLOCK_M, M_out]

    # Add bias
    bias = tl.load(BIAS_ptr + tl.arange(0, M_out), mask=(tl.arange(0, M_out) < M_out), other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + tl.arange(0, M_out)[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (tl.arange(0, M_out)[None, :] < M_out))


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
    Y_flat has shape (M, 3H). Writes:
      B[M, H] = Y[M, :H]
      C[M, H] = Y[M, H:2H]
      XPRJ[M, H] = Y[M, 2H:3H]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + offs_n[None, :] * stride_yN)
    y_vals = tl.load(y_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    # Store B (first H columns)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bM + offs_n[None, :] * stride_bN)
    tl.store(b_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :])

    # Store C (middle H columns, offset H)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cM + offs_n[None, :] * stride_cN)
    tl.store(c_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :])

    # Store XPRJ (last H columns, offset 2H)
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
    Compute grouped causal 1D conv:
      Input X: [M, C_in, L], M=B*S, C_in=H, L=S
      Weight W: [C_in, K], per-channel depthwise
      Output Out: [M, C_in, L]
      groups=C_in means each channel h uses its own W[h, :].
      Default conv1d padding is zero (no padding).
    """
    # We tile over output channels and time positions
    pid_m = tl.program_id(0)  # tile over M (B*S)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)    # [BLOCK_C]
    offs_t = tl.arange(0, BLOCK_T)                      # [BLOCK_T]

    m_mask = offs_m < M
    c_mask = offs_c < C_in
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Loop over output time positions in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # [BLOCK_T]
        ti_mask = t < L  # [BLOCK_T]

        # For each kernel position k, compute input time t_in = t - k (no padding), mask out-of-bounds
        for k in range(0, K):
            t_in = t - k  # [BLOCK_T]
            valid = (t_in >= 0) & (t_in < L) & ti_mask  # [BLOCK_T]

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            # Note: X is contiguous in (M, C, L) layout with strides (stride_xM, stride_xC, stride_xL)
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]

            # Accumulate: acc += x_vals * w_vals[:, None]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_vals = tl.load(BIAS_ptr + offs_c, mask=c_mask, other=0.0)  # [BLOCK_C]
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
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block as (BLOCK_K, OUT_N), W is (OUT_H, IN_H)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wm + offs_n[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x, tl.trans(w))  # [BLOCK_M, BLOCK_N]

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (M_out=3H, H)
        in_proj_bias: (M_out,)
        conv_weight: (C_in=H, K=4)
        conv_bias: (C_in,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        # Ensure contiguous tensors
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        M = B * S
        M_out = 3 * H  # in_proj output channels

        # 1) First linear: y_flat = F.linear(x, in_proj_weight, in_proj_bias)
        # Shapes: X_flat (M, H), W (M_out, H), OUT (M, M_out)
        X_flat = x.view(M, H).contiguous()
        Y_flat = torch.empty((M, 3 * H), dtype=X_flat.dtype, device=X_flat.device)

        # Launch in_proj_linear_kernel
        BLOCK_M = 128
        BLOCK_K = 32
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_K))
        in_proj_linear_kernel[grid_in](
            X_flat, in_proj_weight, in_proj_bias, Y_flat,
            M, H, M_out,
            X_flat.stride(0), X_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            Y_flat.stride(0), Y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # 2) Split y_flat along dim=1 into B, C, x_proj, each (M, H)
        # Output pointers (each of shape (M, H))
        B_mat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)
        C_mat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)
        XPRJ_mat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)

        # Launch chunk_dim1_3_kernel
        BLOCK_M_chunk = 128
        BLOCK_N_chunk = 64
        grid_chunk = (triton.cdiv(M, BLOCK_M_chunk), triton.cdiv(H, BLOCK_N_chunk))
        chunk_dim1_3_kernel[grid_chunk](
            Y_flat, B_mat, C_mat, XPRJ_mat,
            M, H,
            Y_flat.stride(0), Y_flat.stride(1),
            B_mat.stride(0), B_mat.stride(1),
            C_mat.stride(0), C_mat.stride(1),
            XPRJ_mat.stride(0), XPRJ_mat.stride(1),
            BLOCK_M=BLOCK_M_chunk, BLOCK_N=BLOCK_N_chunk,
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((M, H), dtype=B_mat.dtype, device=B_mat.device)
        mul_elementwise_kernel[grid_chunk](
            B_mat, XPRJ_mat, Bx,
            M, H,
            B_mat.stride(0), B_mat.stride(1),
            XPRJ_mat.stride(0), XPRJ_mat.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=BLOCK_M_chunk, BLOCK_N=BLOCK_N_chunk,
        )

        # Reshape to (B, S, H)
        Bx = Bx.view(B, S, H)

        # 4) Grouped causal 1D convolution on Bx: (N=M=B*S, C_in=H, L=S, K=4), groups=H
        # Convert Bx to (M, C_in, L)
        Bx_mcl = Bx.view(M, H, S).contiguous()

        # Output conv_out: (M, C_in, L)
        conv_out = torch.empty((M, H, S), dtype=Bx.dtype, device=Bx.device)

        # Launch grouped_causal_conv1d_kernel
        BLOCK_M_conv = 64
        BLOCK_C_conv = 64
        BLOCK_T_conv = 128
        grid_conv = (triton.cdiv(M, BLOCK_M_conv), triton.cdiv(H, BLOCK_C_conv))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_mcl, conv_weight, conv_bias, conv_out,
            M, H, S, 4,  # K=4
            Bx_mcl.stride(0), Bx_mcl.stride(1), Bx_mcl.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=BLOCK_M_conv, BLOCK_C=BLOCK_C_conv, BLOCK_T=BLOCK_T_conv,
        )

        # 5) Output gating: y = C * conv_out
        # C_mat is (M, H). Reshape conv_out to (M, H, S)
        # Then y is (M, H, S)
        C_mch = C_mat.view(M, H, S).contiguous()
        y = torch.empty((M, H, S), dtype=C_mch.dtype, device=C_mch.device)

        # Elementwise multiplication: (M, H, S) * (M, H, S)
        y = (C_mch * conv_out)

        # 6) Final projection: y @ out_proj_weight^T + out_proj_bias
        # y is (M, H), out_proj_weight: (H, H), out_proj_bias: (H,)
        y_flat = y.reshape(M, H).contiguous()
        out = torch.empty((M, H), dtype=y_flat.dtype, device=y_flat.device)

        # Launch final_proj_kernel
        BLOCK_M_f = 128
        BLOCK_N_f = 64
        BLOCK_K_f = 32
        grid_f = (triton.cdiv(M, BLOCK_M_f), triton.cdiv(H, BLOCK_N_f))
        final_proj_kernel[grid_f](
            y_flat, out_proj_weight, out_proj_bias, out,
            M, H, H,
            y_flat.stride(0), y_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M_f, BLOCK_N=BLOCK_N_f, BLOCK_K=BLOCK_K_f,
        )

        # Reshape to (B, S, H)
        out = out.view(B, S, H)
        return out


def run(*args):
    return ModelNew()(*args)
