import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xn,  # X is (M, H)
    stride_wm, stride_wh,  # W is (M_OUT, H)
    stride_om, stride_on,  # OUT is (M, M_OUT)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute OUT[b, m_out] = sum_h X[b, h] * W[m_out, h] + bias[m_out]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < H

        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        for kk in range(BLOCK_K):
            if k0 + kk < H:
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
    stride_ob1m, stride_ob1n,  # OUT1 is (M, C)
    stride_ob2m, stride_ob2n,  # OUT2 is (M, C)
    stride_ob3m, stride_ob3n,  # OUT3 is (M, C)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C], OUTi: [M, C]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    for n0 in range(0, C, BLOCK_N):
        nn = n0 + offs_n
        nn_mask = nn < C

        # Output 1: channel 0..C-1
        in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + nn[None, :] * stride_in)
        vals1 = tl.load(in_ptrs1, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_ob1m + nn[None, :] * stride_ob1n)
        tl.store(out1_ptrs, vals1, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 2: channel C..2C-1
        in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (C + nn[None, :]) * stride_in)
        vals2 = tl.load(in_ptrs2, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_ob2m + nn[None, :] * stride_ob2n)
        tl.store(out2_ptrs, vals2, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 3: channel 2C..3C-1
        in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (2 * C + nn[None, :]) * stride_in)
        vals3 = tl.load(in_ptrs3, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_ob3m + nn[None, :] * stride_ob3n)
        tl.store(out3_ptrs, vals3, mask=m_mask[:, None] & nn_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,  # A and B are (M, N)
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
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
    M, C_in, L, K,  # K=4, padding_left=3, right_pad=0
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

    # Loop over output time positions in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # For each kernel position k, compute input time t_in = t - k (no padding, zero out of bounds)
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

        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        for kk in range(BLOCK_K):
            if k0 + kk < IN_H:
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


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # x: (B, S, H), weights/bias already provided
        B, S, H = x.shape
        K = conv_weight.shape[2]  # 4
        C_in = conv_weight.shape[1]  # H
        assert C_in == H, "Conv weight channels must equal H"
        assert conv_weight.shape[0] == C_in, "Conv weight first dim must equal C_in"
        assert conv_weight.shape[2] == K, "Conv weight last dim must equal kernel_size"
        assert conv_weight.shape[3] == 1, "Conv weight last dim must be groups=1 per channel in grouped conv"
        assert out_proj_weight.shape[0] == H and out_proj_weight.shape[1] == H, "out_proj_weight must be (H, H)"
        assert out_proj_bias.shape[0] == H, "out_proj_bias must be (H,)"

        # Step 1: in_proj linear
        M = B * S
        # Reshape x to (M, H), weight (M_out, H)
        x_flat = x.reshape(M, H).contiguous()
        M_out = 3 * H
        y_flat = torch.empty((M, M_out), device=x.device, dtype=x.dtype)

        in_proj_linear_kernel[(triton.cdiv(M, 128), triton.cdiv(M_out, 128))](  # tile sizes
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
        )

        # Step 2: chunk along dim=1 (channels) to get B, C, x_proj
        C = H
        B_t = torch.empty((M, C), device=x.device, dtype=x.dtype)
        C_t = torch.empty((M, C), device=x.device, dtype=x.dtype)
        XPRJ_t = torch.empty((M, C), device=x.device, dtype=x.dtype)

        chunk_dim1_3_kernel[(triton.cdiv(M, 128), triton.cdiv(C, 64))](  # tile sizes
            y_flat, B_t, C_t, XPRJ_t,
            M, C,
            y_flat.stride(0), y_flat.stride(1),
            B_t.stride(0), B_t.stride(1),
            C_t.stride(0), C_t.stride(1),
            XPRJ_t.stride(0), XPRJ_t.stride(1),
        )

        # Reshape back to (B, S, C)
        B_t = B_t.view(B, S, C)
        C_t = C_t.view(B, S, C)
        XPRJ_t = XPRJ_t.view(B, S, C)

        # Step 3: element-wise gating
        Bx = torch.empty((B, S, C), device=x.device, dtype=x.dtype)
        mul_elementwise_kernel[(triton.cdiv(B * S, 128), triton.cdiv(C, 64))](  # tile sizes
            B_t, XPRJ_t, Bx,
            B * S, C,
            B_t.stride(0), B_t.stride(2),
            XPRJ_t.stride(0), XPRJ_t.stride(2),
            Bx.stride(0), Bx.stride(2),
        )

        # Step 4: grouped causal 1D convolution: conv over (Bx), groups=H, kernel_size=4, padding=(3,0)
        # Input (M, C_in, L) = (B*S, H, S)
        Bx_contig = Bx.contiguous()
        conv_out = torch.empty((B * S, C_in, S), device=x.device, dtype=x.dtype)

        grouped_causal_conv1d_kernel[(triton.cdiv(B * S, 64), triton.cdiv(C_in, 64), triton.cdiv(S, 128))](  # tile sizes
            Bx_contig, conv_weight, conv_bias, conv_out,
            B * S, C_in, S, K,
            Bx_contig.stride(0), Bx_contig.stride(1), Bx_contig.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        )

        # Step 5: output gating: y = C * conv_out (B, H, S)
        # conv_out is (B*S, H, S), reshape to (B, S, H, S) and broadcast? Better to keep (B*S, H, S) and gate per (b,h)
        # We gate each (b,h) slice: y[b,h,:] = C[b,:,h] * conv_out[b,h,:]
        # First, reshape C_t to (B, S, H) and conv_out to (B, S, H, S) is not correct. We need conv_out (B*S, H, S).
        # Let's compute y as (B, H, S) then transpose later. We’ll compute y_full (B*S, H, S) using broadcasting.

        # For each (b, s, h): y[b,s,h] = C[b,s,h] * conv_out[(b*S + s), h, s]
        # We can compute this with a Triton kernel that maps (m=M=B*S, h=0..H-1, s=0..S-1)
        # But to keep single kernel, we’ll do it with PyTorch for simplicity here, or write another Triton kernel.
        # To strictly adhere to Triton-only, we can write a simple Triton kernel that computes y_full (B*S, H, S) using indexing.

        # Define a Triton kernel to compute y_full = C * conv_out
        # C_t is (B, S, H), conv_out is (B*S, H, S). We need to index C_t[b, s, h] for each m=B*S.

        B_C_t = C_t  # (B, S, H)
        y_full = torch.empty((B * S, C_in, S), device=x.device, dtype=x.dtype)

        # Triton kernel for elementwise multiply along last dimension for (B*S, H, S)
        @triton.jit
        def gate_3d_kernel(
            C_ptr, Out_ptr, Y_ptr,
            B, S, H,
            stride_cm, stride_cn, stride_cs,  # C_t strides for (B,S,H)
            stride_om, stride_on, stride_os,  # conv_out strides for (B*S, H, S)
            stride_ym, stride_yn, stride_ys,  # Y strides for (B*S, H, S)
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
        ):
            pid_m = tl.program_id(0)  # over B*S
            pid_n = tl.program_id(1)  # over H
            pid_t = tl.program_id(2)  # over S

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

            m_mask = offs_m < (B * S)
            n_mask = offs_n < H
            t_mask = offs_t < S

            # Map m to (b, s): b = m // S, s = m % S
            b = offs_m // S
            s = offs_m % S

            # Load C[b, s, n]
            c_ptrs = C_ptr + (b[:, None] * stride_cm + s[:, None] * stride_cn + offs_n[None, :] * stride_cs)
            c_vals = tl.load(c_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

            # Load Out[m, n, t]
            out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on + offs_t[None, :] * stride_os)
            out_vals = tl.load(out_ptrs, mask=m_mask[:, None] & n_mask[None, :] & t_mask[None, :], other=0.0)

            y_vals = c_vals * out_vals  # elementwise multiply

            # Store Y[m, n, t]
            y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn + offs_t[None, :] * stride_ys)
            tl.store(y_ptrs, y_vals, mask=m_mask[:, None] & n_mask[None, :] & t_mask[None, :])

        gate_3d_kernel[(triton.cdiv(B * S, 128), triton.cdiv(H, 64), triton.cdiv(S, 128))](  # tile sizes
            C_t, conv_out, y_full,
            B, S, H,
            C_t.stride(0), C_t.stride(1), C_t.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y_full.stride(0), y_full.stride(1), y_full.stride(2),
        )

        # Step 6: final projection: y -> out_proj(y), y is (B*S, H, S)
        # Flatten y_full to (M, H) where M=B*S, H=H, and apply out_proj_weight (H, H), bias out_proj_bias (H,).
        # We’ll reshape y_full to (M, H) and launch a linear kernel.

        M_final = B * S
        y_for_linear = y_full.view(M_final, H).contiguous()
        output = torch.empty((M_final, H), device=x.device, dtype=x.dtype)

        final_proj_kernel[(triton.cdiv(M_final, 128), triton.cdiv(H, 128))](  # tile sizes
            y_for_linear, out_proj_weight, out_proj_bias, output,
            M_final, H, H,
            y_for_linear.stride(0), y_for_linear.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1),
        )

        # Reshape back to (B, S, H)
        output = output.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
