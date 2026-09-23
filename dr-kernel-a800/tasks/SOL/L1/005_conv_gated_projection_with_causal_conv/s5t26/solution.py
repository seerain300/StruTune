import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # X_ptr: [M, H], W_ptr: [M_OUT, H], OUT_ptr: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    # Loop over K = H dimension
    for k0 in range(0, H, BLOCK_H):
        k = k0 + offs_n
        k_mask = k < H

        # Load X block: [BLOCK_M, BLOCK_H]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_H, BLOCK_H] (we want W[k, :], but we load across n dimension)
        # Note: we compute acc += sum_k x[:, k] * W[n, k]
        # So for each k, we broadcast W[n, k] across rows of x
        # We need W[n, k], but our indexing is W[m_out, h]. Here n = offs_n is m_out index.
        w_ptrs = W_ptr + (n[None, :] * stride_wm + k[:, None] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        # Accumulate: acc += x[:, None, :] * w[None, :, :] -> reduce along k axis
        # Implement outer product accumulation by broadcasting
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=1)

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
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
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Y_ptr: [M, 3H], B_ptr: [M, H], C_ptr: [M, H], XPRJ_ptr: [M, H]
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    h_mask = offs_h < H

    # For B: slice 0..H
    y_b_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + offs_h[None, :] * stride_yH)
    b = tl.load(y_b_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    tl.store(B_ptr + (offs_m[:, None] * stride_bM + offs_h[None, :] * stride_bH), b, mask=m_mask[:, None] & h_mask[None, :])

    # For C: slice H..2H
    y_c_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + (offs_h[None, :] + H) * stride_yH)
    c = tl.load(y_c_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    tl.store(C_ptr + (offs_m[:, None] * stride_cM + offs_h[None, :] * stride_cH), c, mask=m_mask[:, None] & h_mask[None, :])

    # For XPRJ: slice 2H..3H
    y_x_ptrs = Y_ptr + (offs_m[:, None] * stride_yM + (offs_h[None, :] + 2 * H) * stride_yH)
    xprj = tl.load(y_x_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    tl.store(XPRJ_ptr + (offs_m[:, None] * stride_xM + offs_h[None, :] * stride_xH), xprj, mask=m_mask[:, None] & h_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_bh,
    stride_xm, stride_xh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # B_ptr: [M, H], XPRJ_ptr: [M, H], OUT_ptr: [M, H]
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    h_mask = offs_h < H

    b = tl.load(B_ptr + (offs_m[:, None] * stride_bm + offs_h[None, :] * stride_bh), mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    xprj = tl.load(XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh), mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    out = b * xprj
    tl.store(OUT_ptr + (offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh), out, mask=m_mask[:, None] & h_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
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

    # For each output time position t in tiles, accumulate over kernel window
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # Loop over kernel size K
        for k in range(0, K):
            t_in = t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t]
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

        # Load W block: [BLOCK_N, BLOCK_K] (we want W[n, k])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # acc += x @ w^T, i.e., sum over k
        acc += tl.sum(x[:, :, None] * w[None, :, :], axis=2)

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        x: (B, S, H)
        in_proj_weight: (M_out, H), M_out = 3*H
        in_proj_bias: (M_out,)
        conv_weight: (H, K), K=4
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be CUDA for Triton kernels."

        B, S, H = x.shape
        M = B * S

        # 1) Triple linear projection: y = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        # We implement in_proj_linear_kernel for X_flat and W of shape (M_out, H), output (M, M_out).
        # Reshape x to (M, H), then apply kernel, and reshape output to (B, S, 3H).
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, 3 * H), device=x.device, dtype=x.dtype)

        # Launch in_proj_linear_kernel
        BLOCK_M = 128
        BLOCK_H = 128
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(3 * H, BLOCK_H))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, 3 * H,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        )

        # Convert flat y to (B, S, 3H)
        y = y_flat.reshape(B, S, 3 * H).contiguous()

        # 2) Element-wise gating: Bx = B * x_proj
        # Split y along dim=1 (channels): B = y[:, :, :H], C = y[:, :, H:2H], x_proj = y[:, :, 2H:3H], each (B, S, H).
        B_t = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        C_t = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        XPRJ = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        # Chunk kernel along dim=1
        BLOCK_M_chunk = 128
        BLOCK_H_chunk = 64
        grid_chunk = (triton.cdiv(B * S, BLOCK_M_chunk), triton.cdiv(H, BLOCK_H_chunk))
        chunk_dim1_3_kernel[grid_chunk](
            y, B_t, C_t, XPRJ,
            B * S, H,
            y.stride(0), y.stride(2),
            B_t.stride(0), B_t.stride(2),
            C_t.stride(0), C_t.stride(2),
            XPRJ.stride(0), XPRJ.stride(2),
            BLOCK_M=BLOCK_M_chunk, BLOCK_H=BLOCK_H_chunk,
        )

        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        # Elementwise multiply kernel
        BLOCK_M_mul = 128
        BLOCK_H_mul = 64
        grid_mul = (triton.cdiv(B * S, BLOCK_M_mul), triton.cdiv(H, BLOCK_H_mul))
        mul_elementwise_kernel[grid_mul](
            B_t, XPRJ, Bx,
            B * S, H,
            B_t.stride(0), B_t.stride(2),
            XPRJ.stride(0), XPRJ.stride(2),
            Bx.stride(0), Bx.stride(2),
            BLOCK_M=BLOCK_M_mul, BLOCK_H=BLOCK_H_mul,
        )

        # 3) Grouped causal 1D convolution: F.conv1d(Bx, conv_weight, conv_bias, groups=H, kernel_size=4)
        # Pad not used here: PyTorch default padding=0 for conv1d. Implement zero padding in Triton.
        # Input X: (M=B*S, C_in=H, L=S); Weight W: (C_in=H, K=4); Bias: (H,)
        Bx_flat = Bx.reshape(M, H).contiguous()
        Out_conv = torch.empty((M, H), device=x.device, dtype=x.dtype)

        # Triton conv kernel launch
        BLOCK_M_conv = 128
        BLOCK_C_conv = 64
        BLOCK_T_conv = 128
        grid_conv = (triton.cdiv(M, BLOCK_M_conv), triton.cdiv(H, BLOCK_C_conv))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_flat, conv_weight, conv_bias, Out_conv,
            M, H, S, 4,
            Bx_flat.stride(0), Bx_flat.stride(1), Bx_flat.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=BLOCK_M_conv, BLOCK_C=BLOCK_C_conv, BLOCK_T=BLOCK_T_conv,
        )

        # Convert back to (B, H, S)
        conv_out = Out_conv.reshape(B, H, S).contiguous()

        # 4) Output gating: y = C * conv_out
        # C_t: (B, S, H)
        C_t_ = C_t
        y_gated = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        # Elementwise multiply across (B, S, H) with conv_out
        # We can implement a simple elementwise kernel or use PyTorch here since C_t is small.
        # For consistency, implement Triton elementwise kernel to avoid any PyTorch ops.
        B_times = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        # Elementwise kernel: C_t * conv_out
        # We can reuse mul_elementwise_kernel by swapping roles. Define a wrapper:
        grid_gate = (triton.cdiv(B * S, BLOCK_M_mul), triton.cdiv(H, BLOCK_H_mul))
        mul_elementwise_kernel[grid_gate](
            C_t_, conv_out, y_gated,
            B * S, H,
            C_t_.stride(0), C_t_.stride(2),
            conv_out.stride(0), conv_out.stride(2),
            y_gated.stride(0), y_gated.stride(2),
            BLOCK_M=BLOCK_M_mul, BLOCK_H=BLOCK_H_mul,
        )

        # 5) Final output projection: y @ out_proj_weight^T + out_proj_bias, out: (B, S, H)
        # Reshape y_gated to (M, H), out_proj_weight: (H, H)
        y_gated_flat = y_gated.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)

        # Launch final_proj_kernel
        BLOCK_M_final = 128
        BLOCK_N_final = 64
        BLOCK_K_final = 64
        grid_final = (triton.cdiv(M, BLOCK_M_final), triton.cdiv(H, BLOCK_N_final))
        final_proj_kernel[grid_final](
            y_gated_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            y_gated_flat.stride(0), y_gated_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_M_final, BLOCK_N=BLOCK_N_final, BLOCK_K=BLOCK_K_final,
        )

        out = out_flat.reshape(B, S, H).contiguous()
        return out


def run(*args):
    return ModelNew()(*args)
