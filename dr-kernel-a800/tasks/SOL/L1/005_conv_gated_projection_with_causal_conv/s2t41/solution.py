import torch
import triton
import triton.language as tl

# 1) Triple linear projection: given x[B, S, H], compute three outputs (B, S, H)
#    out = x @ W^T + bias, where W is one of the 3 groups of (H, H).
@triton.jit
def triple_linear_bsH_kernel(
    x_ptr,        # *f32, input as (B*S, H) flattened, but we use strides
    W_ptr, b_ptr, # *f32, (H,H) and (H,)
    out_ptr,      # *f32, output (B*S, H)
    B, S, H,
    x_stride0, x_stride1,        # x has logical shape (B*S, H)
    W_stride0, W_stride1,        # W is (H, H)
    out_stride0, out_stride1,    # out is (B*S, H)
    BLOCK_S: tl.constexpr,       # tile along S
    BLOCK_K: tl.constexpr        # tile along H for reduction
):
    pid_bs = tl.program_id(0)  # ranges over B*S
    pid_h = tl.program_id(1)   # ranges over tiles of H
    h_offsets = pid_h * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_h = h_offsets < H

    # Accumulator for this (b*s, h tile)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Iterate over K=H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load x row for this (b*s), across K tile
        x_row_ptrs = x_ptr + pid_bs * x_stride0 + k_offsets * x_stride1
        x_vals = tl.load(x_row_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_K,)

        # Load W[h, k] tile
        W_ptrs = W_ptr + h_offsets[:, None] * W_stride0 + k_offsets[None, :] * W_stride1
        W_vals = tl.load(W_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # shape (BLOCK_K, BLOCK_K)

        # Accumulate: acc[h] += sum_k (x[k] * W[h, k])
        acc += tl.sum(W_vals * x_vals[None, :], axis=1)

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    # Store output
    out_ptrs = out_ptr + pid_bs * out_stride0 + h_offsets * out_stride1
    tl.store(out_ptrs, acc, mask=mask_h)

# 2) Element-wise gating: Bx = B * X, output (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    B_ptr, X_ptr, Out_ptr,
    B, S, H,
    B_stride_b, B_stride_s, B_stride_h,
    X_stride_b, X_stride_s, X_stride_h,
    Out_stride_b, Out_stride_s, Out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    B_ptrs = B_ptr + b * B_stride_b + s_offsets * B_stride_s + h * B_stride_h
    X_ptrs = X_ptr + b * X_stride_b + s_offsets * X_stride_s + h * X_stride_h
    Out_ptrs = Out_ptr + b * Out_stride_b + s_offsets * Out_stride_s + h * Out_stride_h

    B_vals = tl.load(B_ptrs, mask=mask_s, other=0.0)
    X_vals = tl.load(X_ptrs, mask=mask_s, other=0.0)
    Out_vals = B_vals * X_vals
    tl.store(Out_ptrs, Out_vals, mask=mask_s)

# 3) Grouped causal 1D convolution: input Bx (B, S, H), weight (H, H, 4), bias (H), output (B, H, S)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convB_ptr, out_ptr,
    B, S, H,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride0, convW_stride1, convW_stride2,  # (H, H, 4)
    out_stride_b, out_stride_h, out_stride_s,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)   # output channel == input channel since grouped
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for this (b, c, s tile)
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over k in {0,1,2,3}
    # Note: conv_out[b, c, s] = sum_{k=0..3} Bx[b, c, s + k - 1] * convW[c, c, k] + convB[c]
    for k in range(4):
        s_in = s_offsets + k - 1
        # Mask for valid input positions
        mask_in = (s_in >= 0) & (s_in < S) & mask_s

        Bx_ptrs = Bx_ptr + b * Bx_stride_b + s_in * Bx_stride_s + c * Bx_stride_h
        Bx_vals = tl.load(Bx_ptrs, mask=mask_in, other=0.0)

        w = tl.load(convW_ptr + c * convW_stride0 + c * convW_stride1 + k * convW_stride2)
        acc += Bx_vals * w

    # Add bias
    b_c = tl.load(convB_ptr + c)
    acc += b_c

    # Store output
    out_ptrs = out_ptr + b * out_stride_b + c * out_stride_h + s_offsets * out_stride_s
    tl.store(out_ptrs, acc, mask=mask_s)

# 4) Final linear projection: y[B, S, H] -> out[B, S, H] using out_proj_weight (H, H), bias (H)
@triton.jit
def final_linear_bsh_kernel(
    y_ptr, outW_ptr, outB_ptr, out_ptr,
    B, S, H,
    y_stride_b, y_stride_s, y_stride_h,
    outW_stride0, outW_stride1,  # (H, H)
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc_h = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, H, BLOCK_H):
        k_offsets = k0 + tl.arange(0, BLOCK_H)
        mask_k = k_offsets < H

        # For each s, compute dot over K
        for s_i in range(0, BLOCK_S):
            # Gather y[b, s_offsets[s_i], k_offsets]
            y_ptrs = y_ptr + b * y_stride_b + (s_offsets[s_i] * y_stride_s) + k_offsets * y_stride_h
            y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_H,)
            # outW[k, h] = outW_ptr + k*outW_stride0 + h*outW_stride1
            outW_ptrs = outW_ptr + k_offsets[:, None] * outW_stride0 + h * outW_stride1
            outW_vals = tl.load(outW_ptrs, mask=mask_k[:, None], other=0.0)  # shape (BLOCK_H,)

            acc_h += tl.sum(y_vals[:, None] * outW_vals, axis=0)  # sum over K

    # Add bias
    b_h = tl.load(outB_ptr + h)
    acc_h += b_h

    # Store output
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc_h, mask=mask_s)

class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        """
        x: (B, S, H)
        in_proj_weight: (3*H, H)
        in_proj_bias: (3*H)
        conv_weight: (H, H, 4)
        conv_bias: (H)
        out_proj_weight: (H, H)
        out_proj_bias: (H)
        """
        device = x.device
        dtype = x.dtype

        B, S, H = x.shape
        # 1) Triple linear projection: produce B, C, X, each (B, S, H)
        # Prepare (B*S, H) view for kernels
        x_flat = x.reshape(B * S, H).contiguous()

        # B group
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()
        B_out_flat = torch.empty((B * S, H), device=device, dtype=torch.float32)
        grid0 = (B * S, (H + 63) // 64)
        triple_linear_bsH_kernel[grid0](
            x_flat, W0, b0, B_out_flat,
            B, S, H,
            x_flat.stride(0), x_flat.stride(1),
            W0.stride(0), W0.stride(1),
            B_out_flat.stride(0), B_out_flat.stride(1),
            BLOCK_S=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )
        B_out = B_out_flat.reshape(B, S, H)

        # C group
        W1 = in_proj_weight[H:2 * H, :].contiguous()
        b1 = in_proj_bias[H:2 * H].contiguous()
        C_out_flat = torch.empty((B * S, H), device=device, dtype=torch.float32)
        grid1 = (B * S, (H + 63) // 64)
        triple_linear_bsH_kernel[grid1](
            x_flat, W1, b1, C_out_flat,
            B, S, H,
            x_flat.stride(0), x_flat.stride(1),
            W1.stride(0), W1.stride(1),
            C_out_flat.stride(0), C_out_flat.stride(1),
            BLOCK_S=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )
        C_out = C_out_flat.reshape(B, S, H)

        # X group
        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b2 = in_proj_bias[2 * H:3 * H].contiguous()
        X_out_flat = torch.empty((B * S, H), device=device, dtype=torch.float32)
        grid2 = (B * S, (H + 63) // 64)
        triple_linear_bsH_kernel[grid2](
            x_flat, W2, b2, X_out_flat,
            B, S, H,
            x_flat.stride(0), x_flat.stride(1),
            W2.stride(0), W2.stride(1),
            X_out_flat.stride(0), X_out_flat.stride(1),
            BLOCK_S=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )
        X_out = X_out_flat.reshape(B, S, H)

        # 2) Element-wise gating
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

        # 3) Grouped causal conv1d: conv_out[b, c, s], groups=H, kernel_size=4
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
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

        # 4) Output gating
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

        # 5) Final linear projection: y -> output (B, S, H)
        y_flat = y.reshape(B * S, H).contiguous()  # (B*S, H)
        outW = out_proj_weight.contiguous()        # (H, H)
        outB = out_proj_bias.contiguous()          # (H,)
        out_flat = torch.empty((B * S, H), device=device, dtype=torch.float32)
        grid5 = (B * S, (H + 63) // 64)
        final_linear_bsh_kernel[grid5](
            y_flat, outW, outB, out_flat,
            B, S, H,
            y_flat.stride(0), y_flat.stride(1), y_flat.stride(2),
            outW.stride(0), outW.stride(1),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            BLOCK_S=128, BLOCK_H=64,
            num_warps=4, num_stages=2
        )
        out = out_flat.reshape(B, S, H)
        return out


def run(*args):
    return ModelNew()(*args)
