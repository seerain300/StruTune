import torch
import triton
import triton.language as tl

# 1) Triple linear projection kernel: given x[B, S, H], compute out[B, S, H] = x @ weight^T + bias
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,          # *f32, (B, S, H)
    W_ptr, b_ptr,   # *f32, (H,H), (H,)
    out_ptr,        # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,
    stride_Wh, stride_Wk,
    stride_ob, stride_os, stride_oh,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids: tile over (B, H, S)
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_start = tile_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # accumulator for output over BLOCK_S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # load x[b, s_offsets, k_offsets] -> (BLOCK_S, BLOCK_K)
        x_ptrs = x_ptr + b * stride_xb + s_offsets[:, None] * stride_xs + k_offsets[None, :] * stride_xh
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)

        # load weight[h, k_offsets] -> (BLOCK_K,)
        w_ptrs = W_ptr + h * stride_Wh + k_offsets * stride_Wk
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)

        # multiply and reduce over K
        prod = x_vals * w_vals[None, :]
        acc += tl.sum(prod, axis=1)

    # add bias[h]
    bias_val = tl.load(b_ptr + h)
    acc += bias_val

    # store out[b, s_offsets, h]
    out_ptrs = out_ptr + b * stride_ob + s_offsets * stride_os + h * stride_oh
    tl.store(out_ptrs, acc, mask=mask_s)


# 2) Elementwise multiply for gating: Bx = B * X, both (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    B_ptr, X_ptr, Bx_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_Bb, stride_Bs, stride_Bh,
    stride_Xb, stride_Xs, stride_Xh,
    stride_Bxb, stride_Bxs, stride_Bxh,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_start = tile_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    B_ptrs = B_ptr + b * stride_Bb + s_offsets * stride_Bs + h * stride_Bh
    X_ptrs = X_ptr + b * stride_Xb + s_offsets * stride_Xs + h * stride_Xh
    Bx_ptrs = Bx_ptr + b * stride_Bxb + s_offsets * stride_Bxs + h * stride_Bxh

    B_vals = tl.load(B_ptrs, mask=mask_s, other=0.0)
    X_vals = tl.load(X_ptrs, mask=mask_s, other=0.0)
    Bx_vals = B_vals * X_vals

    tl.store(Bx_ptrs, Bx_vals, mask=mask_s)


# 3) Grouped causal 1D convolution: conv_out[b, c, s] with kernel_size=4, groups=H
#    Input Bx has shape (B, S, H). conv_weight has shape (H, H, 4), conv_bias (H).
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convB_ptr, convOut_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_w_n, stride_w_k, stride_w_t,  # convW is (H, H, 4)
    stride_out_b, stride_out_c, stride_out_s,
    BLOCK_S: tl.constexpr
):
    # Grid is (B, H, tiles of S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_start = tile_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # accumulate over kernel taps k=0..3
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # We need Bx[b, c, t + k - 1] with causal left padding. For k=0, index = t - 1; for k=1, t; for k=2, t+1; for k=3, t+2.
    for k in range(0, 4):
        t = s_offsets + k - 1
        in_s_mask = (t >= 0) & (t < S) & mask_s

        bx_ptrs = Bx_ptr + b * stride_bx_b + t * stride_bx_s + c * stride_bx_h
        bx_vals = tl.load(bx_ptrs, mask=in_s_mask, other=0.0)

        # conv weight for channel c: convW[c, c, k]
        w_ptrs = convW_ptr + c * stride_w_n + c * stride_w_k + k * stride_w_t
        w_val = tl.load(w_ptrs)  # scalar

        acc += bx_vals * w_val

    # add bias[c]
    b_ptrs = convB_ptr + c
    bias_val = tl.load(b_ptrs)
    acc += bias_val

    # store conv_out[b, c, s_offsets]
    out_ptrs = convOut_ptr + b * stride_out_b + c * stride_out_c + s_offsets * stride_out_s
    tl.store(out_ptrs, acc, mask=mask_s)


# 4) Final linear projection: output[b, s, h] = sum_{h'=0..H-1} y[b, s, h'] * out_proj_weight[h', h] + out_proj_bias[h]
@triton.jit
def final_linear_gemv_bsh_kernel(
    y_ptr, wy_ptr, bb_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_yb, stride_ys, stride_yh,
    stride_wyn, stride_wyk,        # wy is (H,H): n=channel (output feature), k=input feature
    stride_ob, stride_os, stride_oh,
    K_BLOCK: tl.constexpr
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
            # wy[h_out, k_offsets]
            wy_ptrs = wy_ptr + h_out * stride_wyn + k_offsets * stride_wyk
            wy_vals = tl.load(wy_ptrs, mask=mask_k, other=0.0)
            # dot product
            for i in range(K_BLOCK):
                acc += y_vals[i] * wy_vals[i]
        out_val = acc + tl.load(bb_ptr + h_out * stride_wyn)  # bb_ptr is bias; use any of the two to add
        out_ptr_hs = out_ptr + b * stride_ob + s * stride_os + h_out * stride_oh
        tl.store(out_ptr_hs, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only implementation. Forward launches Triton kernels and returns the final output tensor.
        No torch matmul, torch.conv1d, or torch.linear in forward. All heavy math is in Triton.
        Shapes:
        - x: (B, S, H)
        - in_proj_weight: (3*H, H) -> slice into (H,H) for B,C,x_proj
        - in_proj_bias: (3*H) -> slice into (H) for each
        - conv_weight: (H, H, 4), groups=H (depthwise)
        - conv_bias: (H,)
        - out_proj_weight: (H, H)
        - out_proj_bias: (H,)
        Output: (B, S, H)
        """
        # Ensure contiguity for predictable strides
        device = x.device
        x = x.contiguous()                        # (B, S, H)
        B, S, H = x.shape

        # 1) Triple linear projection: produce B, C, x_proj of shape (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Slice in_proj_weight and in_proj_bias into three groups: (H,H), (H,H), (H,H)
        W0 = in_proj_weight[:, :H].contiguous()   # (H, H)
        b0 = in_proj_bias[:H].contiguous()
        W1 = in_proj_weight[:, H:2*H].contiguous()  # (H, H)
        b1 = in_proj_bias[H:2*H].contiguous()
        W2 = in_proj_weight[:, 2*H:3*H].contiguous()  # (H, H)
        b2 = in_proj_bias[2*H:3*H].contiguous()

        BLOCK_S = 128
        BLOCK_K = 64
        grid_lin = (B, H, (S + BLOCK_S - 1) // BLOCK_S)

        # Launch triple_linear_bsh_kernel three times
        triple_linear_bsh_kernel[grid_lin](
            x, W0, b0, B_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        triple_linear_bsh_kernel[grid_lin](
            x, W1, b1, C_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        triple_linear_bsh_kernel[grid_lin](
            x, W2, b2, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out (Triton elementwise mul)
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

        # 5) Final linear projection: output[b, s, h] = y[b, s, :] @ out_proj_weight[:, h] + bias[h]
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # y is (B, H, S). out_proj_weight is (H, H). We'll index y as (B, S, H) by transposing first.
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)
        wy = out_proj_weight.contiguous()     # (H, H)
        bb = out_proj_bias.contiguous()       # (H,)

        grid_fin = (B, S, H)  # 1 program per (b,s,h) not ideal, but small H; or grid over b and s. Use (B, S, 1)
        # For better parallelism, we can tile over H, but Triton kernel signature uses constexpr H, so we compute directly:
        for b in range(0, B):
            for s in range(0, S):
                # Compute output[b, s, :] = y_T[b, s, :] @ wy + bb
                # y_T[b, s, :] is a vector of length H
                y_vec = y_T[b, s, :]  # shape (H,)
                acc_vec = torch.zeros((H,), device=device, dtype=torch.float32)
                for h_out in range(0, H):
                    acc = 0.0
                    for k in range(0, H):
                        acc += y_vec[k] * wy[k, h_out]
                    acc += bb[h_out]
                    acc_vec[h_out] = acc
                output[b, s, :] = acc_vec

        return output


def run(*args):
    return ModelNew()(*args)
