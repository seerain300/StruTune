import torch
import triton
import triton.language as tl

# 1) Triple linear projection: given x[B, S, H], compute
#    B[B,S,H] = x @ W0^T + b0, C[B,S,H] = x @ W1^T + b1, X[B,S,H] = x @ W2^T + b2
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,     # *f32, (B, S, H)
    W0_ptr, b0_ptr,  # *f32, (H,H), (H,)
    W1_ptr, b1_ptr,  # *f32, (H,H), (H,)
    W2_ptr, b2_ptr,  # *f32, (H,H), (H,)
    B_out_ptr, C_out_ptr, X_out_ptr,  # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,
    stride_W0n, stride_W0k,
    stride_W1n, stride_W1k,
    stride_W2n, stride_W2k,
    stride_Bb, stride_Bs, stride_Bh,
    stride_Cb, stride_Cs, stride_Ch,
    stride_Xb, stride_Xs, stride_Xh,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)  # batch
    h_out = tl.program_id(1)  # feature channel
    s_block = tl.program_id(2)  # block along seq

    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulators
    accB = tl.zeros((BLOCK_S,), dtype=tl.float32)
    accC = tl.zeros((BLOCK_S,), dtype=tl.float32)
    accX = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over input feature k in [0, H)
    for k in range(0, H):
        # Load x[b, s, k] for this block
        x_ptrs = x_ptr + b * stride_xb + s_offsets * stride_xs + k * stride_xh
        x_vals = tl.load(x_ptrs, mask=mask_s, other=0.0)  # (BLOCK_S,)

        # Weight0[h_out, k]
        w0_val = tl.load(W0_ptr + h_out * stride_W0n + k * stride_W0k)
        accB += x_vals * w0_val

        # Weight1[h_out, k]
        w1_val = tl.load(W1_ptr + h_out * stride_W1n + k * stride_W1k)
        accC += x_vals * w1_val

        # Weight2[h_out, k]
        w2_val = tl.load(W2_ptr + h_out * stride_W2n + k * stride_W2k)
        accX += x_vals * w2_val

    # Add biases
    b0_val = tl.load(b0_ptr + h_out)
    b1_val = tl.load(b1_ptr + h_out)
    b2_val = tl.load(b2_ptr + h_out)

    accB += b0_val
    accC += b1_val
    accX += b2_val

    # Store results
    B_out_ptrs = B_out_ptr + b * stride_Bb + s_offsets * stride_Bs + h_out * stride_Bh
    C_out_ptrs = C_out_ptr + b * stride_Cb + s_offsets * stride_Cs + h_out * stride_Ch
    X_out_ptrs = X_out_ptr + b * stride_Xb + s_offsets * stride_Xs + h_out * stride_Xh

    tl.store(B_out_ptrs, accB, mask=mask_s)
    tl.store(C_out_ptrs, accC, mask=mask_s)
    tl.store(X_out_ptrs, accX, mask=mask_s)


# 2) Elementwise multiply: gate = A * B, elementwise (Triton)
@triton.jit
def elemwise_mul_bsh_kernel(
    A_ptr, B_ptr, Out_ptr,  # *f32
    Bsz, S, H,
    stride_Ab, stride_As, stride_Ah,
    stride_Bb, stride_Bs, stride_Bh,
    stride_Ob, stride_Os, stride_Oh,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)

    s_offsets = s_block * 128 + tl.arange(0, 128)
    mask = s_offsets < S

    A_ptrs = A_ptr + b * stride_Ab + s_offsets * stride_As + h * stride_Ah
    B_ptrs = B_ptr + b * stride_Bb + s_offsets * stride_Bs + h * stride_Bh
    Out_ptrs = Out_ptr + b * stride_Ob + s_offsets * stride_Os + h * stride_Oh

    a = tl.load(A_ptrs, mask=mask, other=0.0)
    b_val = tl.load(B_ptrs, mask=mask, other=0.0)
    tl.store(Out_ptrs, a * b_val, mask=mask)


# 3) Grouped causal 1D convolution:
#    Input Bx: (B, H, S) -> conv with groups=H, kernel_size=4 -> conv_out (B, H, S)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *f32, (B, H, S)
    convW_ptr, convB_ptr,   # *f32, convW: (H, H, 4), convB: (H,)
    out_ptr,        # *f32, (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_Bxb, stride_Bxh, stride_Bxs,
    stride_wgn, stride_wgk, stride_wgt,   # convW strides: n=channel, k=hidden, t=kernel pos
    stride_ob, stride_oh, stride_os,
):
    b = tl.program_id(0)
    c = tl.program_id(1)  # channel in groups
    s_block = tl.program_id(2)

    s_offsets = s_block * 128 + tl.arange(0, 128)
    mask_s = s_offsets < S

    acc = tl.zeros((128,), dtype=tl.float32)

    # Kernel size = 4, causal: shift = t - (t + k - 1) => valid when 0 <= t + k - 1 < S
    for k in range(0, 4):
        # valid positions: t + k - 1 in [0, S-1]
        in_pos = s_offsets + (k - 1)
        valid = (in_pos >= 0) & (in_pos < S)
        # Load Bx[b, c, in_pos] with mask
        Bx_ptrs = Bx_ptr + b * stride_Bxb + c * stride_Bxh + in_pos * stride_Bxs
        val = tl.load(Bx_ptrs, mask=mask_s & valid, other=0.0)
        # Load conv_weight[c, c, k]
        w_ptr = convW_ptr + c * stride_wgn + c * stride_wgk + k * stride_wgt
        w_val = tl.load(w_ptr)
        acc += val * w_val

    # Add bias
    bias_ptr = convB_ptr + c
    bias_val = tl.load(bias_ptr)
    acc += bias_val

    # Store to out[b, c, s_offsets]
    out_ptrs = out_ptr + b * stride_ob + c * stride_oh + s_offsets * stride_os
    tl.store(out_ptrs, acc, mask=mask_s)


# 4) Final linear projection: given y[B,S,H], compute out[B,S,H] = y @ outW^T + bias
@triton.jit
def final_linear_gemv_bsh_kernel(
    y_ptr,         # *f32, (B, S, H)
    outW_ptr, outB_ptr,    # *f32, (H, H), (H,)
    out_ptr,       # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_yb, stride_ys, stride_yh,
    stride_wyn, stride_wyk,    # outW is (H,H): n=channel (output feature), k=input feature
    stride_ob, stride_os, stride_oh,
    K_BLOCK: tl.constexpr,     # tile size over H
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    for h_out in range(0, H):
        acc = 0.0
        for k in range(0, H, K_BLOCK):
            k_offsets = k + tl.arange(0, K_BLOCK)
            mask_k = k_offsets < H
            # y[b, s, k_offsets]
            y_ptrs = y_ptr + b * stride_yb + s * stride_ys + k_offsets * stride_yh
            y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # (K_BLOCK,)
            # outW[h_out, k_offsets] -> (K_BLOCK,)
            wy_ptrs = outW_ptr + h_out * stride_wyn + k_offsets * stride_wyk
            wy_vals = tl.load(wy_ptrs, mask=mask_k, other=0.0)
            # dot product accumulate
            for i in range(K_BLOCK):
                acc += y_vals[i] * wy_vals[i]
        out_val = acc + tl.load(outB_ptr + h_out * stride_oh)
        out_ptr_hs = out_ptr + b * stride_ob + s * stride_os + h_out * stride_oh
        tl.store(out_ptr_hs, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only forward. Launches three Triton kernels:
        1) Triple linear projection -> B, C, x_proj
        2) Elementwise multiply (B * x_proj) -> Bx
        3) Grouped causal conv1d -> conv_out
        4) Elementwise multiply (C * conv_out) -> y
        5) Final linear projection -> output
        No torch matmul/conv/linear in forward. All heavy math is in Triton kernels.
        """
        # Input: x (B, S, H), in_proj_weight (3*H, H), in_proj_bias (3*H),
        # conv_weight (H, H, 4), conv_bias (H), out_proj_weight (H, H), out_proj_bias (H)
        B, S, H = x.shape

        # 1) Triple linear projection -> (B,S,H) for B, C, x_proj
        B_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        C_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        X_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        # Slice in_proj_weight and in_proj_bias into three groups: (H,H), (H,H), (H,H)
        W0 = in_proj_weight[:, :H].contiguous()  # (H, H)
        b0 = in_proj_bias[:H].contiguous()
        W1 = in_proj_weight[:, H:2 * H].contiguous()  # (H, H)
        b1 = in_proj_bias[H:2 * H].contiguous()
        W2 = in_proj_weight[:, 2 * H:3 * H].contiguous()  # (H, H)
        b2 = in_proj_bias[2 * H:3 * H].contiguous()

        BLOCK_S = 256
        grid_triple = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        triple_linear_bsh_kernel[grid_triple](
            x, W0, b0, W1, b1, W2, b2,
            B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out[b, c, s] with kernel_size=4, groups=H
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B, H, S)
        y = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y (B,H,S) -> out (B,S,H) with out_proj (H,H)
        out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        outW = out_proj_weight.contiguous()  # (H,H)
        outB = out_proj_bias.contiguous()    # (H,)
        grid_final = (B, S, H)
        final_linear_gemv_bsh_kernel[grid_final](
            y, outW, outB, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            outW.stride(0), outW.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            K_BLOCK=64,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
