import torch
import triton
import triton.language as tl

@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_out,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X_ptr: [M, H], W_ptr: [M_out, H], OUT_ptr: [M, M_out]
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over M_out (output channels)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of X/OUT: (B*S, M_out)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels (M_out)
    offs_k = tl.arange(0, BLOCK_K)                    # reduction over H

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < H
        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K] (W: [M_out, H])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(x, w)

    # Add bias: bias[offs_n]
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Store results to OUT
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    Y_ptr, B_ptr, C_ptr, XPRJ_ptr,
    M, H,
    stride_yM, stride_yH,
    stride_bM, stride_bH,
    stride_cM, stride_cH,
    stride_xM, stride_xH,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Y_ptr: [M, 3H] (M = B*S), we read three slices: B [:H], C [H:2H], x_proj [2H:3H]
    # Write out B, C, x_proj each of shape [M, H]
    pid_m = tl.program_id(0)  # tile over M
    pid_k = tl.program_id(1)  # tile over H (we will run grid over H too)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    k_mask = offs_k < H

    # Load Y: Y[M, 3H]
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + (offs_k[None, :] + 0) * stride_yH)  # for B
    y0 = tl.load(y_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]
    tl.store(B_ptr + (offs_m[:, None] * stride_bM + offs_k[None, :] * stride_bH), y0, mask=m_mask[:, None] & k_mask[None, :])

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + (offs_k[None, :] + H) * stride_yH)  # for C
    y1 = tl.load(y_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]
    tl.store(C_ptr + (offs_m[:, None] * stride_cM + offs_k[None, :] * stride_cH), y1, mask=m_mask[:, None] & k_mask[None, :])

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + (offs_k[None, :] + 2 * H) * stride_yH)  # for x_proj
    y2 = tl.load(y_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]
    tl.store(XPRJ_ptr + (offs_m[:, None] * stride_xM + offs_k[None, :] * stride_xH), y2, mask=m_mask[:, None] & k_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bM, stride_bH,
    stride_xM, stride_xH,
    stride_oM, stride_oH,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute Bx = B * x_proj, each shape (M, H)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    k_mask = offs_k < H

    b_ptrs = B_ptr + (offs_m[:, None] * stride_bM + offs_k[None, :] * stride_bH)
    x_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xM + offs_k[None, :] * stride_xH)
    b = tl.load(b_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    out = b * x

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_oM + offs_k[None, :] * stride_oH)
    tl.store(out_ptrs, out, mask=m_mask[:, None] & k_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] (we pass M=B*S, C_in=H, L=S)
    # W_ptr: [C_in, K]
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # For each output time position t, accumulate over kernel window
    # conv1d default padding=0, groups=C_in
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # Loop over kernel size K
        for k in range(0, K):
            # causal: t_in = t - k
            t_in = t - k
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
            # Accumulate: acc += x_vals * w_vals[:, None]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c  # bias[c] per output channel
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
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H
        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K] (W: [OUT_H, IN_H])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, w)

    # Add bias: bias[offs_n]
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Store results to OUT
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
        # x: (B, S, H), all weights/biases as PyTorch tensors
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]  # 3*H
        K = conv_weight.shape[2]         # kernel_size

        # 1) Triple linear projection: x -> (B, S, 3H) via in_proj
        # Flatten (B, S, H) -> (M, H), M = B*S
        M = B * S
        X = x.reshape(M, H)
        # Allocate output (M, M_out)
        Y = torch.empty((M, M_out), dtype=torch.float32, device=x.device)

        # Launch in_proj_linear_kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid](
            X, in_proj_weight, in_proj_bias, Y,
            M, H, M_out,
            X.stride(0), X.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Reshape Y to (B, S, 3H)
        Y = Y.view(B, S, M_out)

        # 2) Split into B, C, x_proj along channel dim=1
        # We implement slicing via a Triton kernel that writes B, C, x_proj from Y_flat
        # First flatten Y to (M, 3H)
        Y_flat = Y.reshape(M, 3 * H)  # M=B*S
        B_out = torch.empty((M, H), dtype=torch.float32, device=x.device)
        C_out = torch.empty((M, H), dtype=torch.float32, device=x.device)
        x_proj_out = torch.empty((M, H), dtype=torch.float32, device=x.device)

        BLOCK_M2 = 256
        BLOCK_K2 = 64
        grid_chunk = (triton.cdiv(M, BLOCK_M2), triton.cdiv(H, BLOCK_K2))
        chunk_dim1_3_kernel[grid_chunk](
            Y_flat, B_out, C_out, x_proj_out,
            M, H,
            Y_flat.stride(0), Y_flat.stride(1),
            B_out.stride(0), B_out.stride(1),
            C_out.stride(0), C_out.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_K=BLOCK_K2,
        )

        # Reshape back to (B, S, H)
        B_t = B_out.view(B, S, H)
        C_t = C_out.view(B, S, H)
        x_proj_t = x_proj_out.view(B, S, H)

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_mul = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        mul_elementwise_kernel[grid_mul](
            B_t, x_proj_t, Bx,
            M, H,
            B_t.stride(0), B_t.stride(1),
            x_proj_t.stride(0), x_proj_t.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=128, BLOCK_K=64,
        )

        # 4) Grouped causal 1D convolution: Bx -> (B, H, S) with groups=H, kernel_size=4, padding=0
        # Prepare X_conv as (M, H, S)
        X_conv = Bx.reshape(M, H, S)
        # Allocate output (M, H, S)
        conv_out = torch.empty((M, H, S), dtype=torch.float32, device=x.device)

        # Launch grouped_causal_conv1d_kernel
        BLOCK_Mc = 128
        BLOCK_Cc = 64
        BLOCK_Tc = 128
        grid_conv = (triton.cdiv(M, BLOCK_Mc), triton.cdiv(H, BLOCK_Cc))
        grouped_causal_conv1d_kernel[grid_conv](
            X_conv, conv_weight, conv_bias, conv_out,
            M, H, S, K,
            X_conv.stride(0), X_conv.stride(1), X_conv.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=BLOCK_Mc, BLOCK_C=BLOCK_Cc, BLOCK_T=BLOCK_Tc,
        )

        # 5) Output gating: y = C * conv_out, shape (B, H, S)
        y_pre = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
        # Reshape to (M, H, S) then elementwise multiply
        y_pre = (C_t * conv_out).view(B, H, S)

        # 6) Final output projection: y @ out_proj_weight^T + out_proj_bias
        # Flatten y_pre to (M, H)
        y_flat = y_pre.reshape(M, H)
        out_flat = torch.empty((M, H), dtype=torch.float32, device=x.device)

        # Launch final_proj_kernel
        BLOCK_Mf = 128
        BLOCK_Nf = 64
        BLOCK_Kf = 64
        grid_final = (triton.cdiv(M, BLOCK_Mf), triton.cdiv(H, BLOCK_Nf))
        final_proj_kernel[grid_final](
            y_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            y_flat.stride(0), y_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_Mf, BLOCK_N=BLOCK_Nf, BLOCK_K=BLOCK_Kf,
        )

        # Reshape to (B, S, H)
        output = out_flat.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
