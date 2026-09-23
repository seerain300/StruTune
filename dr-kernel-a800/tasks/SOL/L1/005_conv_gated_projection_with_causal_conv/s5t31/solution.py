import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
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
        # Load X block: [BLOCK_M, BLOCK_K]
        in_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xn)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K] (note: we want W[n, k])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wn)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        # x: [BM, BK], w: [BN, BK] -> we need w^T: [BK, BN]
        acc += tl.dot(x, tl.trans(w))

    # Add bias: BIAS[n] per output channel
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
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
    in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * 0) * stride_in)
    out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n)
    val1 = tl.load(in_ptrs1, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out1_ptrs, val1, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 2 (middle C_IN): base = C_IN
    in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * C_IN) * stride_in)
    out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_o2m + offs_n[None, :] * stride_o2n)
    val2 = tl.load(in_ptrs2, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out2_ptrs, val2, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 3 (last C_IN): base = 2*C_IN
    in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * (2 * C_IN)) * stride_in)
    out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_o3m + offs_n[None, :] * stride_o3n)
    val3 = tl.load(in_ptrs3, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out3_ptrs, val3, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    IN1_ptr, IN2_ptr, OUT_ptr,
    M, N,
    stride_i1m, stride_i1n,
    stride_i2m, stride_i2n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN1: [M, N], IN2: [M, N], OUT: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a = tl.load(IN1_ptr + offs_m[:, None] * stride_i1m + offs_n[None, :] * stride_i1n, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(IN2_ptr + offs_m[:, None] * stride_i2m + offs_n[None, :] * stride_i2n, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = a * b

    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=m_mask[:, None] & n_mask[None, :])


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
    # W_ptr: [C_in, K] (groups=C_in)
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # For each output time position t in tiles, accumulate over kernel window
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # Loop over kernel size K with zero padding (no right-pad)
        for k in range(0, K):
            t_in = t - k  # causal
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
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

        # Load W block: [BLOCK_N, BLOCK_K] (note: we want W[n, k])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias: BIAS[n] per output channel
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized fused layer:
        1. in_proj: F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        2. chunk into B, C, x_proj
        3. element-wise gating: Bx = B * x_proj
        4. grouped causal conv1d with kernel_size=4, groups=H, padding=0
        5. output gating: y = C * conv_out
        6. final projection: y @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        """
        assert x.ndim == 3, "x must be (B, S, H)"
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]
        assert M_out == 3 * H, "in_proj_weight must have (3*H) rows"

        device = x.device
        # Prepare shapes and strides
        # Flatten batch and seq for GEMM convenience: M = B*S
        M = B * S
        IN_H = H
        OUT_H_1 = M_out  # for in-proj
        C_in_conv = H    # channels for conv
        K = conv_weight.shape[2]  # kernel_size
        OUT_H_2 = H     # output channels of conv (groups=H)
        OUT_H_final = H  # final projection output H

        # 1) in_proj: compute y_flat = X_flat @ W^T + bias
        # X_flat: [M, H] => x reshaped
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, OUT_H_1), device=device, dtype=x.dtype)

        # Launch in_proj_linear_kernel
        # Choose blocks
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(OUT_H_1, BLOCK_N))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, IN_H, OUT_H_1,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) y has shape (B, S, 3H); we get it by viewing y_flat as (B, S, 3H)
        # Create a 3D tensor view and then chunk along dim=1
        # We avoid torch.chunk by using a custom kernel that writes chunked outputs
        y3 = y_flat.view(B, S, 3 * H)  # (B, S, 3H)
        # Allocate outputs for B, C, x_proj
        B_out = torch.empty((M, H), device=device, dtype=x.dtype)  # (B*S, H)
        C_out = torch.empty((M, H), device=device, dtype=x.dtype)  # (B*S, H)
        x_proj_out = torch.empty((M, H), device=device, dtype=x.dtype)  # (B*S, H)

        # Launch chunk_dim1_3_kernel on y3 viewed as (M, 3*C_in) where C_in=H
        # We need strides for y3 and outputs
        # y3 strides: for (B, S, 3H) contiguous, stride along first dim is S*3H, along second is 3H, along third is 1
        stride_im = y3.stride(0)  # S*3H
        stride_in = y3.stride(1)  # 3H
        stride_o1m = B_out.stride(0)  # M (B*S)
        stride_o1n = B_out.stride(1)  # H
        stride_o2m = C_out.stride(0)
        stride_o2n = C_out.stride(1)
        stride_o3m = x_proj_out.stride(0)
        stride_o3n = x_proj_out.stride(1)

        BLOCK_M_c = 64
        BLOCK_N_c = 32
        grid_chunk = (triton.cdiv(M, BLOCK_M_c), triton.cdiv(H, BLOCK_N_c))
        chunk_dim1_3_kernel[grid_chunk](
            y3, B_out, C_out, x_proj_out,
            M, H,
            stride_im, stride_in,
            stride_o1m, stride_o1n,
            stride_o2m, stride_o2n,
            stride_o3m, stride_o3n,
            BLOCK_M=BLOCK_M_c, BLOCK_N=BLOCK_N_c,
        )

        # 3) element-wise gating
        Bx = torch.empty((M, H), device=device, dtype=x.dtype)  # (B*S, H)
        mul_elementwise_kernel[(triton.cdiv(M, 64), triton.cdiv(H, 32))](  # grid for (M, H)
            B_out, x_proj_out, Bx,
            M, H,
            B_out.stride(0), B_out.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=64, BLOCK_N=32
        )

        # Reshape Bx back to (B, S, H)
        Bx3 = Bx.view(B, S, H)

        # 4) grouped causal conv1d with kernel_size=4, groups=H, padding=0
        # We need to pass Bx as (M, C_in, L) = (B*S, H, S). We already have Bx3; we flatten and set strides as if (M, H, S)
        # Here, M=B*S, C_in=H, L=S. conv_weight: (C_in, K). conv_bias: (C_in)
        # Allocate conv_out: (M, C_in, L)
        conv_out = torch.empty((M, C_in_conv, S), device=device, dtype=x.dtype)

        stride_xM = Bx3.reshape(M, H, S).stride(0)  # should be H*S
        stride_xC = Bx3.reshape(M, H, S).stride(1)  # should be S
        stride_xL = Bx3.reshape(M, H, S).stride(2)  # should be 1

        stride_wC = conv_weight.stride(0)  # C_in
        stride_wK = conv_weight.stride(1)  # K

        stride_oM = conv_out.stride(0)  # M
        stride_oC = conv_out.stride(1)  # C_in
        stride_oL = conv_out.stride(2)  # L

        BLOCK_M_conv = 64
        BLOCK_C_conv = 64
        BLOCK_T_conv = 64
        grid_conv = (triton.cdiv(M, BLOCK_M_conv), triton.cdiv(C_in_conv, BLOCK_C_conv))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx3.reshape(M, H, S), conv_weight, conv_bias, conv_out,
            M, C_in_conv, S, K,
            stride_xM, stride_xC, stride_xL,
            stride_wC, stride_wK,
            stride_oM, stride_oC, stride_oL,
            BLOCK_M=BLOCK_M_conv, BLOCK_C=BLOCK_C_conv, BLOCK_T=BLOCK_T_conv,
        )

        # conv_out is (B*S, H, S); reshape to (B, H, S) for next gating
        conv_out_3d = conv_out.view(B, H, S)

        # 5) Output gating: y = C * conv_out, (B, H, S)
        # We can perform elementwise multiplication here. To stay Triton-only, use mul_elementwise_kernel on 2D reshapes:
        y_before_final = torch.empty((B * H * S), device=device, dtype=x.dtype)
        # Reshape C_out to (B*S, H) and conv_out_3d to (B*S, H) by flattening over S with H groups
        # Flatten (B, H, S) to (B*S, H) for multiplication:
        # conv_out_3d_flat: (B*S, H)
        conv_out_flat = conv_out_3d.reshape(B * H, S)  # (B*H, S)
        # We need (B*S, H); so we do: (B*S, H) is conv_out_3d.view(B, H, S) -> (B*S, H) by stacking over S
        # Better: conv_out_3d.view(B, H, S) -> (B*S, H) is conv_out_3d.reshape(B*S, H) but conv_out_3d is (B, H, S)
        # To get (B*S, H), we can use conv_out_3d.view(B, H, S) then flatten second dim: (B*S, H)
        # Let's recompute a flat version directly from conv_out_3d:
        conv_out_flat = conv_out_3d.view(B, H, S).reshape(B * S, H)  # (B*S, H)
        C_flat = C_out.view(B * S, H)  # (B*S, H)

        # Launch mul_elementwise_kernel
        y_before_final = torch.empty((B * S, H), device=device, dtype=x.dtype)
        mul_elementwise_kernel[(triton.cdiv(B * S, 64), triton.cdiv(H, 32))](  # grid for (B*S, H)
            C_flat, conv_out_flat, y_before_final,
            B * S, H,
            C_flat.stride(0), C_flat.stride(1),
            conv_out_flat.stride(0), conv_out_flat.stride(1),
            y_before_final.stride(0), y_before_final.stride(1),
            BLOCK_M=64, BLOCK_N=32
        )

        # 6) Final projection: y @ out_proj_weight^T + out_proj_bias
        # y_before_final: (B*S, H), out_proj_weight: (H, H), out_proj_bias: (H)
        y_proj = torch.empty((B * S, H), device=device, dtype=x.dtype)

        BLOCK_M_f = 64
        BLOCK_N_f = 64
        BLOCK_K_f = 32
        grid_final = (triton.cdiv(B * S, BLOCK_M_f), triton.cdiv(H, BLOCK_N_f))
        final_proj_kernel[grid_final](
            y_before_final, out_proj_weight, out_proj_bias, y_proj,
            B * S, H, H,
            y_before_final.stride(0), y_before_final.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            y_proj.stride(0), y_proj.stride(1),
            BLOCK_M=BLOCK_M_f, BLOCK_N=BLOCK_N_f, BLOCK_K=BLOCK_K_f,
        )

        # Reshape to (B, S, H)
        output = y_proj.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
