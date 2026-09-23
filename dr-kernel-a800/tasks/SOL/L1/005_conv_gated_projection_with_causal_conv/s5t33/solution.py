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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X: [M, H], W: [M_OUT, H], OUT: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_N):
        k = k0 + offs_n  # columns in W/OUT
        k_mask = k < H

        # Load X block: [BLOCK_M, BLOCK_N]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_M] but we want [BLOCK_M, BLOCK_N] -> transpose
        w_ptrs = W_ptr + (k[None, :] * stride_wm + offs_m[:, None] * stride_wh)
        w = tl.load(w_ptrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0)

        # acc += x @ w
        acc += tl.dot(x, tl.trans(w))

    # Add bias: BIAS[n] per output channel
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    IN_ptr, OUT1_ptr, OUT2_ptr, OUT3_ptr,
    M, C_IN,
    stride_im, stride_in,
    stride_o1m, stride_o1n,
    stride_o2m, stride_o2n,
    stride_o3m, stride_o3n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C_IN] where M = B*S
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over C_IN (each output corresponds to one of 3 chunks)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_IN

    # For chunk 1 (first C_IN): base = 0
    in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n)
    val1 = tl.load(in_ptrs1, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out1_ptrs, val1, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 2 (middle C_IN): base = C_IN
    in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] + C_IN) * stride_in)
    out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_o2m + offs_n[None, :] * stride_o2n)
    val2 = tl.load(in_ptrs2, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out2_ptrs, val2, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 3 (last C_IN): base = 2*C_IN
    in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] + 2 * C_IN) * stride_in)
    out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_o3m + offs_n[None, :] * stride_o3n)
    val3 = tl.load(in_ptrs3, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out3_ptrs, val3, mask=m_mask[:, None] & n_mask[None, :])


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

    # For each output position t in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # For each kernel position k in {0..K-1}, compute t_in = t - k (no padding, zero out of bounds)
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

        # Load W block: [BLOCK_N, BLOCK_K] (note: W is [OUT_H, IN_H], so we index n over rows and k over cols)
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # acc += x @ w (x: [BM,BK], w: [BK,BN] -> [BM,BN])
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        # Ensure tensors are on CUDA and float32
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda \
               and conv_weight.is_cuda and conv_bias.is_cuda \
               and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be on CUDA"

        B, S, H = x.shape
        # 1) First linear: y = F.linear(x, in_proj_weight, in_proj_bias), M_out = 3*H
        M = B * S
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, 3 * H), device=x.device, dtype=x.dtype)  # output of linear
        M_OUT = 3 * H

        # Launch in_proj_linear_kernel
        BLOCK_M = 128
        BLOCK_N = 64
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_OUT, BLOCK_N))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_OUT,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 2) Chunk along dim=1 to get B, C, x_proj (each shape [M, H])
        C_IN = H
        y_flat = y_flat.contiguous()
        B_out = torch.empty((M, C_IN), device=x.device, dtype=x.dtype)
        C_out = torch.empty((M, C_IN), device=x.device, dtype=x.dtype)
        xproj_out = torch.empty((M, C_IN), device=x.device, dtype=x.dtype)

        # Launch chunk_dim1_3_kernel
        BLOCK_M_c = 128
        BLOCK_N_c = 64
        grid_chunk = (triton.cdiv(M, BLOCK_M_c), triton.cdiv(C_IN, BLOCK_N_c))
        chunk_dim1_3_kernel[grid_chunk](
            y_flat, B_out, C_out, xproj_out,
            M, C_IN,
            y_flat.stride(0), y_flat.stride(1),
            B_out.stride(0), B_out.stride(1),
            C_out.stride(0), C_out.stride(1),
            xproj_out.stride(0), xproj_out.stride(1),
            BLOCK_M=BLOCK_M_c, BLOCK_N=BLOCK_N_c,
        )

        # Reshape back to (B, S, H)
        B_mat = B_out.reshape(B, S, H).contiguous()
        C_mat = C_out.reshape(B, S, H).contiguous()
        xproj_mat = xproj_out.reshape(B, S, H).contiguous()

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        # Launch mul_elementwise_kernel over (B,S,H)
        BLOCK_M_m = 128
        BLOCK_N_m = 64
        M_m = B * S
        Bx_flat = Bx.reshape(M_m, H)
        b_flat = B_mat.reshape(M_m, H)
        xproj_flat = xproj_mat.reshape(M_m, H)

        grid_mul = (triton.cdiv(M_m, BLOCK_M_m), triton.cdiv(H, BLOCK_N_m))
        mul_elementwise_kernel[grid_mul](
            b_flat, xproj_flat, Bx_flat,
            M_m, H,
            b_flat.stride(0), b_flat.stride(1),
            xproj_flat.stride(0), xproj_flat.stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=BLOCK_M_m, BLOCK_N=BLOCK_N_m,
        )
        Bx = Bx_flat.reshape(B, S, H).contiguous()

        # 4) Grouped causal 1D convolution: conv over (B, H, S), groups=H, kernel_size=4, padding=0
        # Represent Bx as (M, H, S)
        Bx_flat = Bx.reshape(M, H, S).contiguous()
        conv_weight = conv_weight.contiguous()  # [C_in, K], here K=4, C_in=H
        conv_bias = conv_bias.contiguous()
        conv_out = torch.empty((M, C_IN, S), device=x.device, dtype=x.dtype)

        # Launch grouped_causal_conv1d_kernel
        BLOCK_M_conv = 128
        BLOCK_C_conv = 64
        BLOCK_T_conv = 128
        grid_conv = (triton.cdiv(M, BLOCK_M_conv), triton.cdiv(C_IN, BLOCK_C_conv))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_flat, conv_weight, conv_bias, conv_out,
            M, C_IN, S, 4,  # K=4
            Bx_flat.stride(0), Bx_flat.stride(1), Bx_flat.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=BLOCK_M_conv, BLOCK_C=BLOCK_C_conv, BLOCK_T=BLOCK_T_conv,
        )
        conv_out = conv_out.reshape(B, C_IN, S).contiguous()

        # 5) Output gating: y = C * conv_out
        y_causal = torch.empty((B, C_IN, S), device=x.device, dtype=x.dtype)
        M2 = B * S  # B*C_IN
        # Prepare input for final projection: shape [B*S, C_IN]
        y_causal_flat = y_causal.reshape(M2, C_IN).contiguous()
        C_flat = C_mat.reshape(M2, C_IN).contiguous()
        conv_out_flat = conv_out.reshape(M2, C_IN).contiguous()

        grid_mul2 = (triton.cdiv(M2, BLOCK_M_m), triton.cdiv(C_IN, BLOCK_N_m))
        mul_elementwise_kernel[grid_mul2](
            C_flat, conv_out_flat, y_causal_flat,
            M2, C_IN,
            C_flat.stride(0), C_flat.stride(1),
            conv_out_flat.stride(0), conv_out_flat.stride(1),
            y_causal_flat.stride(0), y_causal_flat.stride(1),
            BLOCK_M=BLOCK_M_m, BLOCK_N=BLOCK_N_m,
        )
        y_causal = y_causal_flat.reshape(B, C_IN, S).contiguous()

        # 6) Final projection: y_causal @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        out = torch.empty((B * S, H), device=x.device, dtype=x.dtype)
        y_flat_final = y_causal.reshape(M, H).contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        grid_final = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        final_proj_kernel[grid_final](
            y_flat_final, out_proj_weight, out_proj_bias, out,
            M, H, H,
            y_flat_final.stride(0), y_flat_final.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=64,
        )
        return out.reshape(B, S, H)


def run(*args):
    return ModelNew()(*args)
