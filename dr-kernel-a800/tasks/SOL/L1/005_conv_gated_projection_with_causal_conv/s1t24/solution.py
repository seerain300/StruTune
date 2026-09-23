import torch
import triton
import triton.language as tl


# 1) Triton linear projection: out[B, S, M] = x @ W[:M, :].T + bias[:M]
# x: (B, S, H), W: (M, H), out: (B, S, M)
@triton.jit
def TritonLinearKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    m = pid_m

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros([BLOCK_S, BLOCK_M], dtype=tl.float32)

    # Reduction over H (input feature dimension)
    for k_start in range(0, H, BLOCK_K):
        h_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_h = h_offsets < H

        # Load X tile: shape [BLOCK_S, BLOCK_K]
        x_ptrs = x_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + h_offsets[None, :] * stride_x_h
        x_tile = tl.load(x_ptrs, mask=(s_offsets[:, None] < S) & mask_h[None, :], other=0.0).to(tl.float32)

        # Load W tile: shape [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + m_offsets[:, None] * stride_w_m + h_offsets[None, :] * stride_w_h
        w_tile = tl.load(w_ptrs, mask=(m_offsets[:, None] < M) & mask_h[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += X @ W^T
        # x_tile: [S, K], w_tile: [M, K] -> w_tile^T: [K, M]
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Add bias: bias is [M], broadcast over S
    bias_ptrs = bias_ptr + m_offsets
    bias_vals = tl.load(bias_ptrs, mask=(m_offsets < M), other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Store output: out[b, s, m]
    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m_offsets[None, :] * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets[:, None] < S) & (m_offsets[None, :] < M))


# 2) Triton element-wise gating: Bx = B * x_proj
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# 3) Triton left-pad along sequence by PAD=3 for causal conv
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD]
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiles over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # Write zeros for the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # Copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# 4) Triton grouped causal 1D convolution with groups=H, kernel_size=4:
# Input: Bx_pad of shape (B, H, S+3); conv_weight: (H, 1, 4); conv_bias: (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_b
    h = pid_h

    # We output across S positions
    for s_out in range(0, BLOCK_S):
        s = s_out  # we will iterate s from 0 to S-1
        # Compute starting index in padded input for causal
        start = s - PAD  # since PAD=3, start = s - 3
        acc = tl.zeros((), dtype=tl.float32)
        # sum over k=0..3 of Bx_pad[b, h, s + k]
        for k in range(0, 4):
            idx = start + k
            in_bounds = (idx >= 0) & (idx < S)
            val = tl.load(Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + idx * stride_bx_s, mask=in_bounds, other=0.0)
            w_ptr = conv_weight_ptr + h * stride_w_h + k * stride_w_k
            w_val = tl.load(w_ptr).to(tl.float32)
            acc += val * w_val

        # add bias
        bias_val = tl.load(conv_bias_ptr + h).to(tl.float32)
        acc += bias_val

        # store to conv_out[b, h, s]
        out_ptr_pos = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        tl.store(out_ptr_pos, acc)


# 5) Triton elementwise multiply: Y[b, s, h] = C[b, s, h] * conv_out[b, h, s]
@triton.jit
def TritonMulKernel(
    C_ptr, conv_out_ptr, Y_ptr,
    B, S, H,
    stride_c_b, stride_c_s, stride_c_h,
    stride_co_b, stride_co_h, stride_co_s,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # Load C: [B, S, H]
    c_ptrs = C_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    c_vals = tl.load(c_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    # Load conv_out: [B, H, S]
    co_ptrs = conv_out_ptr + pid_b * (H * S) + h_offsets[:, None] * H + s_offsets[None, :]
    co_vals = tl.load(co_ptrs, mask=mask_s[None, :] & mask_h[:, None], other=0.0).to(tl.float32)

    y_vals = c_vals * co_vals

    y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(y_ptrs, y_vals, mask=mask_s[:, None] & mask_h[None, :])


# 6) Triton final projection: OUT[b, s, m] = sum_h Y[b, s, h] * W[m, h] + bias[m]
# Y: [B, S, H], W: [M, H] (M=H), bias: [M], OUT: [B, S, M]
@triton.jit
def TritonLinearFinalKernel(
    Y_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    m = pid_m

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros([BLOCK_S, BLOCK_M], dtype=tl.float32)

    # Reduction over H (input feature dimension)
    for k_start in range(0, H, BLOCK_K):
        h_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_h = h_offsets < H

        # Load Y tile: shape [BLOCK_S, BLOCK_K]
        y_ptrs = Y_ptr + b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
        y_tile = tl.load(y_ptrs, mask=(s_offsets[:, None] < S) & mask_h[None, :], other=0.0).to(tl.float32)

        # Load W tile: shape [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + m_offsets[:, None] * stride_w_m + h_offsets[None, :] * stride_w_h
        w_tile = tl.load(w_ptrs, mask=(m_offsets[:, None] < M) & mask_h[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += Y @ W^T
        acc += tl.dot(y_tile, tl.trans(w_tile))

    # Add bias: bias is [M], broadcast over S
    bias_ptrs = bias_ptr + m_offsets
    bias_vals = tl.load(bias_ptrs, mask=(m_offsets < M), other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Store output: out[b, s, m]
    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m_offsets[None, :] * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets[:, None] < S) & (m_offsets[None, :] < M))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Shapes
        Bsz, S, H = x.shape
        # Ensure CUDA tensors for Triton
        device = x.device
        dtype = x.dtype

        # 1) Three linear projections
        # Prepare weights/biases for each group
        W0 = in_proj_weight[:H, :]  # (H, H)
        b0 = in_proj_bias[:H]       # (H,)
        W1 = in_proj_weight[H:2*H, :]  # (H, H)
        b1 = in_proj_bias[H:2*H]      # (H,)
        W2 = in_proj_weight[2*H:3*H, :]  # (H, H)
        b2 = in_proj_bias[2*H:3*H]      # (H,)

        # Allocate outputs
        B = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)
        C = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)
        x_proj = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)

        # Grids
        BLOCK_S = 128
        BLOCK_M = 64
        BLOCK_K = 64

        grid_linear = (Bsz, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_M))

        TritonLinearKernel[grid_linear](
            x, W0, b0, B, Bsz, S, H, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        TritonLinearKernel[grid_linear](
            x, W1, b1, C, Bsz, S, H, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        TritonLinearKernel[grid_linear](
            x, W2, b2, x_proj, Bsz, S, H, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # 2) Element-wise gating
        Bx = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)
        BLOCK_Sg = 128
        BLOCK_Hg = 64
        grid_gate = (Bsz, triton.cdiv(S, BLOCK_Sg), triton.cdiv(H, BLOCK_Hg))
        TritonGateKernel[grid_gate](B, x_proj, Bx, Bsz, S, H, BLOCK_S=BLOCK_Sg, BLOCK_H=BLOCK_Hg)

        # 3) Left-pad along S by PAD=3
        Bx_pad = torch.empty((Bsz, H, S + 3), device=device, dtype=torch.float32)
        BLOCK_Sp = 256
        grid_pad = (Bsz, H, triton.cdiv(S + 3, BLOCK_Sp))
        TritonPadLeftKernel[grid_pad](
            Bx, Bx_pad, Bsz, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=BLOCK_Sp,
        )

        # 4) Grouped causal 1D convolution (groups=H, kernel_size=4)
        # conv_weight: (H, 1, 4)
        conv_out = torch.empty((Bsz, H, S), device=device, dtype=torch.float32)
        BLOCK_Sc = 256
        grid_conv = (Bsz, H)
        TritonGroupedCausalConvKernel[grid_conv](
            Bx_pad, conv_weight, conv_bias, conv_out,
            Bsz, H, S, 3,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_Sc,
        )

        # 5) Output gating: y = C * conv_out (elementwise)
        y = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)
        grid_mul = (Bsz, triton.cdiv(S, BLOCK_Sg), triton.cdiv(H, BLOCK_Hg))
        TritonMulKernel[grid_mul](
            C, conv_out, y,
            Bsz, S, H,
            C.stride(0), C.stride(1), C.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=BLOCK_Sg, BLOCK_H=BLOCK_Hg,
        )

        # 6) Final projection: F.linear(y, out_proj_weight, out_proj_bias)
        # out_proj_weight: (H, H), out_proj_bias: (H)
        # Allocate output
        out = torch.empty((Bsz, S, H), device=device, dtype=torch.float32)
        # Launch TritonLinearFinalKernel
        M = H  # final output has H channels
        BLOCK_Sf = 128
        BLOCK_Mf = 64
        BLOCK_Kf = 64
        grid_final = (Bsz, triton.cdiv(S, BLOCK_Sf), triton.cdiv(M, BLOCK_Mf))
        TritonLinearFinalKernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            Bsz, S, M, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=BLOCK_Sf, BLOCK_M=BLOCK_Mf, BLOCK_K=BLOCK_Kf,
        )

        return out


def run(*args):
    return ModelNew()(*args)
