import torch
import triton
import triton.language as tl


@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,          # *float32, input (B, S, H), contiguous
    in_w0_ptr, in_b0_ptr,    # *float32, *float32, weight and bias for B: (H, H), (H,)
    in_w1_ptr, in_b1_ptr,    # *float32, *float32, weight and bias for C: (H, H), (H,)
    in_w2_ptr, in_b2_ptr,    # *float32, *float32, weight and bias for x_proj: (H, H), (H,)
    outB_ptr, outC_ptr, outX_ptr,  # *float32 outputs (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,
    stride_w0n, stride_w0k, stride_b0,      # B weight strides
    stride_w1n, stride_w1k, stride_b1,      # C weight strides
    stride_w2n, stride_w2k, stride_b2,      # x_proj weight strides
    stride_ob, stride_os, stride_oh,
    BLOCK_S: tl.constexpr,
):
    # Each program computes for a given (b, c_out, s_block)
    b = tl.program_id(0)
    c_out = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulators for three outputs at positions s_offsets
    acc_B = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc_X = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over H (features) in tiles
    for h_k in range(0, H, 64):
        h_offsets = h_k + tl.arange(0, 64)
        mask_h = h_offsets < H

        # Load x tiles: x[b, s_offsets, h_offsets] -> (BLOCK_S, 64)
        x_ptrs = x_ptr + b * stride_xb + s_offsets[:, None] * stride_xs + h_offsets[None, :] * stride_xh
        x_tile = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

        # B group: (H, H)
        w0_ptrs = in_w0_ptr + h_offsets[None, :] * stride_w0n + s_offsets[:, None] * stride_w0k
        w0_tile = tl.load(w0_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0)  # (64, BLOCK_S)
        # acc_B += sum over h of x[b, s, h] * w0[h, c_out]
        # x_tile shape: (BLOCK_S, 64), w0_tile shape: (64, BLOCK_S)
        acc_B += tl.sum(x_tile * w0_tile, axis=1)  # reduce over 64

        # C group: (H, H)
        w1_ptrs = in_w1_ptr + h_offsets[None, :] * stride_w1n + s_offsets[:, None] * stride_w1k
        w1_tile = tl.load(w1_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0)
        acc_C += tl.sum(x_tile * w1_tile, axis=1)

        # x_proj group: (H, H)
        w2_ptrs = in_w2_ptr + h_offsets[None, :] * stride_w2n + s_offsets[:, None] * stride_w2k
        w2_tile = tl.load(w2_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0)
        acc_X += tl.sum(x_tile * w2_tile, axis=1)

    # Add biases: (H,)
    b0 = tl.load(in_b0_ptr + h_offsets, mask=mask_h, other=0.0)  # length 64
    b1 = tl.load(in_b1_ptr + h_offsets, mask=mask_h, other=0.0)
    b2 = tl.load(in_b2_ptr + h_offsets, mask=mask_h, other=0.0)

    # Store results for this c_out
    # outB[b, s_offsets, c_out], outC[b, s_offsets, c_out], outX[b, s_offsets, c_out]
    for i in range(64):
        hi = h_k + i
        if hi < H:
            outB_ptrs = outB_ptr + b * stride_ob + s_offsets * stride_os + hi * stride_oh
            outC_ptrs = outC_ptr + b * stride_ob + s_offsets * stride_os + hi * stride_oh
            outX_ptrs = outX_ptr + b * stride_ob + s_offsets * stride_os + hi * stride_oh
            tl.store(outB_ptrs, acc_B[i], mask=mask_s)
            tl.store(outC_ptrs, acc_C[i], mask=mask_s)
            tl.store(outX_ptrs, acc_X[i], mask=mask_s)


@triton.jit
def grouped_causal_conv1d_kernel(
    bx_ptr,        # *float32, input Bx after causal padding: (B, H, S), contiguous
    conv_w_ptr, conv_b_ptr,      # *float32, *float32, conv_weight: (H, H, 4), conv_bias: (H,)
    conv_out_ptr,                    # *float32 output: (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bxb, stride_bxc, stride_bxs,
    stride_wcn, stride_wck, stride_wck2,   # conv_weight strides: (H, H, 4)
    stride_ob, stride_oc, stride_os,
):
    # Grid: (B, H, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    acc = 0.0

    # Apply grouped depthwise conv with kernel_size=4 and causal padding
    for k in range(4):
        in_t = t + k - 1  # causal: pad zeros on left
        if in_t >= 0 and in_t < S:
            bx_val = tl.load(bx_ptr + b * stride_bxb + c * stride_bxc + in_t * stride_bxs)
            conv_w_val = tl.load(conv_w_ptr + c * stride_wcn + c * stride_wck + k * stride_wck2)
            acc += bx_val * conv_w_val

    # Add bias
    bias_val = tl.load(conv_b_ptr + c * stride_ob)
    acc += bias_val

    # Store
    tl.store(conv_out_ptr + b * stride_ob + c * stride_oc + t * stride_os, acc)


@triton.jit
def final_linear_gemv_bsh_kernel(
    y_ptr,         # *float32, input y: (B, S, H), contiguous
    wy_ptr, bb_ptr,# *float32, *float32, out_proj_weight: (H, H), bias: (H,)
    out_ptr,       # *float32 output: (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_yb, stride_ys, stride_yh,
    stride_wyn, stride_wyk,
    stride_ob, stride_os, stride_oh,
    K_BLOCK: tl.constexpr,
):
    # Grid: (B, S, H)
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = 0.0
    for k in range(0, H, K_BLOCK):
        k_offsets = k + tl.arange(0, K_BLOCK)
        mask_k = k_offsets < H
        # y[b, s, k_offsets]
        y_ptrs = y_ptr + b * stride_yb + s * stride_ys + k_offsets * stride_yh
        y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # (K_BLOCK,)
        # wy[h_out, k_offsets] -> (K_BLOCK,)
        wy_ptrs = wy_ptr + h_out * stride_wyn + k_offsets * stride_wyk
        wy_vals = tl.load(wy_ptrs, mask=mask_k, other=0.0)
        # dot product
        for i in range(K_BLOCK):
            acc += y_vals[i] * wy_vals[i]
    out_val = acc + tl.load(bb_ptr + h_out * stride_oh)
    out_ptr_hs = out_ptr + b * stride_ob + s * stride_os + h_out * stride_oh
    tl.store(out_ptr_hs, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only forward. Launches three Triton kernels to perform:
        1) Triple linear projection (B, C, x_proj) from x.
        2) Grouped causal 1D convolution with kernel_size=4 (groups=hidden_size).
        3) Final linear projection.
        No torch matmul, conv1d, or elementwise gating in forward. All heavy math is in Triton.
        """
        # Ensure contiguity for predictable strides
        x = x.contiguous()                    # (B, S, H)
        H = x.shape[-1]
        B = x.shape[0]
        S = x.shape[1]

        # 1) Triple linear projection: produce B, C, x_proj of shape (B, S, H)
        B_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        C_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        X_out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        # Slice in_proj_weight and in_proj_bias into three groups: (H,H), (H,H), (H,H)
        # in_proj_weight is (3*H, H); per-group slices:
        W0 = in_proj_weight[:, :H].contiguous()   # (H, H)
        b0 = in_proj_bias[:H].contiguous()
        W1 = in_proj_weight[:, H:2*H].contiguous()  # (H, H)
        b1 = in_proj_bias[H:2*H].contiguous()
        W2 = in_proj_weight[:, 2*H:3*H].contiguous()  # (H, H)
        b2 = in_proj_bias[2*H:3*H].contiguous()

        # Launch triple linear kernel
        BLOCK_S = 256
        grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        triple_linear_bsh_kernel[grid](
            x, W0, b0, W1, b1, W2, b2,
            B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1), 0,        # bias strides not used; pass 0
            W1.stride(0), W1.stride(1), 0,
            W2.stride(0), W2.stride(1), 0,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out (implement with Triton elementwise mul)
        B


def run(*args):
    return ModelNew()(*args)
