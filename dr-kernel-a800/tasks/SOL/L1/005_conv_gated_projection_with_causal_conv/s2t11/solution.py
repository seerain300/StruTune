import torch
import triton
import triton.language as tl

# 1) Triple linear projection: given x[B, S, H], compute three outputs (B,S,H)
#    out0 = x @ W0^T + b0, out1 = x @ W1^T + b1, out2 = x @ W2^T + b2
#    where in_proj_weight is (3*H, H), we slice to (H,H) for each group.
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    W0_ptr, b0_ptr,   # *f32, (H,H), (H,)
    W1_ptr, b1_ptr,   # *f32, (H,H), (H,)
    W2_ptr, b2_ptr,   # *f32, (H,H), (H,)
    out0_ptr, out1_ptr, out2_ptr,  # *f32, (B,S,H)
    B, S, H,
    x_stride_b, x_stride_s, x_stride_h,
    W0_stride_n, W0_stride_k,
    W1_stride_n, W1_stride_k,
    W2_stride_n, W2_stride_k,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, H, ceil(S/BLOCK_S))
    b = tl.program_id(0)
    h_out = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Accumulators per output position in the tile
    acc0 = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over hidden dimension k
    for k in range(0, H):
        # Load x[b, s, k] for all s in tile
        x_ptrs = x_ptr + b * x_stride_b + s_offsets * x_stride_s + k * x_stride_h
        x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)

        # Load weight rows for each output (h_out fixed)
        # W0[h_out, k] (scalar)
        w0 = tl.load(W0_ptr + h_out * W0_stride_n + k * W0_stride_k)
        # W1[h_out, k]
        w1 = tl.load(W1_ptr + h_out * W1_stride_n + k * W1_stride_k)
        # W2[h_out, k]
        w2 = tl.load(W2_ptr + h_out * W2_stride_n + k * W2_stride_k)

        # Accumulate
        acc0 += x_vals * w0
        acc1 += x_vals * w1
        acc2 += x_vals * w2

    # Add biases
    b0 = tl.load(b0_ptr + h_out)
    b1 = tl.load(b1_ptr + h_out)
    b2 = tl.load(b2_ptr + h_out)

    acc0 += b0
    acc1 += b1
    acc2 += b2

    # Store results
    out0_ptrs = out0_ptr + b * out_stride_b + s_offsets * out_stride_s + h_out * out_stride_h
    out1_ptrs = out1_ptr + b * out_stride_b + s_offsets * out_stride_s + h_out * out_stride_h
    out2_ptrs = out2_ptr + b * out_stride_b + s_offsets * out_stride_s + h_out * out_stride_h
    tl.store(out0_ptrs, acc0, mask=s_mask)
    tl.store(out1_ptrs, acc1, mask=s_mask)
    tl.store(out2_ptrs, acc2, mask=s_mask)


# 2) Element-wise gating: Bx = B_out * X_out (B,S,H)
@triton.jit
def elemwise_mul_bsh_kernel(
    a_ptr, b_ptr, out_ptr,
    B, S, H,
    a_stride_b, a_stride_s, a_stride_h,
    b_stride_b, b_stride_s, b_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offsets < S
    a_ptrs = a_ptr + b * a_stride_b + s_offsets * a_stride_s + h * a_stride_h
    b_ptrs = b_ptr + b * b_stride_b + s_offsets * b_stride_s + h * b_stride_h
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, a_vals * b_vals, mask=mask)


# 3) Grouped causal 1D convolution (depthwise, groups=H, kernel_size=4):
#    conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t+k-1] * conv_weight[c, c, k] + conv_bias[c]
#    Assumes Bx is (B, H, S) and conv_weight is (H, H, 4), bias (H,).
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *f32, (B, H, S) input for conv
    convW_ptr, convB_ptr,  # *f32, (H,H,4), (H,)
    conv_out_ptr,        # *f32, (B, H, S) output
    B, S, H,
    Bx_stride_b, Bx_stride_c, Bx_stride_s,
    convW_stride_n, convW_stride_k, convW_stride_c,
    conv_out_stride_b, conv_out_stride_c, conv_out_stride_s,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, H, ceil(S/BLOCK_S))
    b = tl.program_id(0)
    c = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # kernel_size=4 causal: t_in = t + k - 1
    for k in range(4):
        t_in = s_offsets + k - 1
        in_mask = (t_in >= 0) & (t_in < S) & mask_s
        # Load Bx[b, c, t_in]
        bx_ptrs = Bx_ptr + b * Bx_stride_b + c * Bx_stride_c + t_in * Bx_stride_s
        bx_vals = tl.load(bx_ptrs, mask=in_mask, other=0.0)
        # Load conv weight convW[c, c, k] (scalar)
        w = tl.load(convW_ptr + c * convW_stride_n + c * convW_stride_k + k * convW_stride_c)
        acc += bx_vals * w

    # Add bias
    bval = tl.load(convB_ptr + c)
    acc += bval

    # Store conv_out[b, c, s]
    out_ptrs = conv_out_ptr + b * conv_out_stride_b + c * conv_out_stride_c + s_offsets * conv_out_stride_s
    tl.store(out_ptrs, acc, mask=mask_s)


# 4) Final linear projection: y[B, S, H] = conv_out @ out_proj_weight^T + out_proj_bias
#    out_proj_weight is (H, H), out_proj_bias is (H,). Output is (B, S, H).
@triton.jit
def final_linear_gemv_bsh_kernel(
    y_ptr,        # *f32, (B, S, H) input
    wy_ptr,       # *f32, (H, H) out_proj_weight
    bb_ptr,       # *f32, (H,) out_proj_bias
    out_ptr,      # *f32, (B, S, H) output
    B, S, H,
    y_stride_b, y_stride_s, y_stride_h,
    wy_stride_n, wy_stride_k,        # wy is (H,H): n=output channel, k=input feature
    out_stride_b, out_stride_s, out_stride_h,
    K_BLOCK: tl.constexpr,
):
    # Grid: (B, S, H) — one program per (b, s, h)
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = 0.0
    # Loop over H in tiles
    for k in range(0, H, K_BLOCK):
        k_offsets = k + tl.arange(0, K_BLOCK)
        mask_k = k_offsets < H

        # y[b, s, k_offsets] -> (K_BLOCK,)
        y_ptrs = y_ptr + b * y_stride_b + s * y_stride_s + k_offsets * y_stride_h
        y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # (K_BLOCK,)

        # wy[h_out, k_offsets] -> (K_BLOCK,)
        wy_ptrs = wy_ptr + h_out * wy_stride_n + k_offsets * wy_stride_k
        wy_vals = tl.load(wy_ptrs, mask=mask_k, other=0.0)

        # Accumulate dot product for this tile
        acc += tl.sum(y_vals * wy_vals, axis=0)

    out_val = acc + tl.load(bb_ptr + h_out)
    out_ptrs = out_ptr + b * out_stride_b + s * out_stride_s + h_out * out_stride_h
    tl.store(out_ptrs, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only forward. Launches three Triton kernels:
        1) triple_linear_bsh_kernel to produce B, C, x_proj.
        2) grouped_causal_conv1d_kernel to compute conv_out from Bx (elementwise B*x_proj).
        3) final_linear_gemv_bsh_kernel to compute final output.
        No torch matmul/conv/elementwise gating in forward. All heavy ops are Triton kernels.
        """
        # Ensure contiguity for predictable strides
        x = x.contiguous()           # (B, S, H)
        B, S, H = x.shape
        device = x.device
        dtype = x.dtype

        # 1) Triple linear projection: produce B, C, X (each B,S,H)
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Slice in_proj_weight (3*H,H) into three (H,H) groups
        W0 = in_proj_weight[:, :H].contiguous()   # (H,H)
        b0 = in_proj_bias[:H].contiguous()        # (H,)
        W1 = in_proj_weight[:, H:2*H].contiguous()  # (H,H)
        b1 = in_proj_bias[H:2*H].contiguous()     # (H,)
        W2 = in_proj_weight[:, 2*H:3*H].contiguous()  # (H,H)
        b2 = in_proj_bias[2*H:3*H].contiguous()   # (H,)

        BLOCK_S = 128
        grid1 = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        triple_linear_bsh_kernel[grid1](
            x, W0, b0, W1, b1, W2, b2,
            B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out[b, c, s] with kernel_size=4, groups=H
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # output is (B,H,S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B,H,S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 5) Final linear projection to (B,S,H)
        out_final = torch.empty((B, S, H), device=device, dtype=torch.float32)
        out_proj_w = out_proj_weight.contiguous()  # (H,H)
        out_proj_b = out_proj_bias.contiguous()    # (H,)
        grid2 = (B, S, H)
        final_linear_gemv_bsh_kernel[grid2](
            y, out_proj_w, out_proj_b, out_final,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w.stride(0), out_proj_w.stride(1),
            out_final.stride(0), out_final.stride(1), out_final.stride(2),
            K_BLOCK=64,
            num_warps=2, num_stages=2
        )

        # Return result as the original model's final output shape: (B, S, H)
        return out_final


def run(*args):
    return ModelNew()(*args)
