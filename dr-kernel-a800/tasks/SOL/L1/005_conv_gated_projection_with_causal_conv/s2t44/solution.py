import torch
import triton
import triton.language as tl


# 1) Triple linear projection: compute B, C, X (each [B, S, H]) from x[B, S, H]
#    Using in_proj_weight slices: W0, W1, W2 each (H,H) and biases (H,)
#    out[b, s, h] = sum_k x[b, s, k] * weight[h, k] + bias[h]
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,     # *f32, (B, S, H)
    W0_ptr, b0_ptr,   # *f32, (H,H), (H,)
    W1_ptr, b1_ptr,   # *f32, (H,H), (H,)
    W2_ptr, b2_ptr,   # *f32, (H,H), (H,)
    B_out_ptr, C_out_ptr, X_out_ptr,  # *f32, (B,S,H)
    B, S, H,
    x_stride_b, x_stride_s, x_stride_h,
    W0_stride0, W0_stride1,
    W1_stride0, W1_stride1,
    W2_stride0, W2_stride1,
    B_out_stride_b, B_out_stride_s, B_out_stride_h,
    C_out_stride_b, C_out_stride_s, C_out_stride_h,
    X_out_stride_b, X_out_stride_s, X_out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)

    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulators for three outputs
    acc0 = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over K dimension (input hidden size)
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load x[b, s_offsets, k_offsets] as a [BLOCK_S, BLOCK_K] matrix
        x_ptrs = x_ptr + b * x_stride_b + s_offsets[:, None] * x_stride_s + k_offsets[None, :] * x_stride_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_S, BLOCK_K]

        # Load corresponding weights for output channel h
        w0_ptrs = W0_ptr + h * W0_stride0 + k_offsets * W0_stride1  # (BLOCK_K,)
        w1_ptrs = W1_ptr + h * W1_stride0 + k_offsets * W1_stride1  # (BLOCK_K,)
        w2_ptrs = W2_ptr + h * W2_stride0 + k_offsets * W2_stride1  # (BLOCK_K,)

        w0 = tl.load(w0_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        w1 = tl.load(w1_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        w2 = tl.load(w2_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Accumulate: sum over K
        # acc += sum_k x_vals[:, k] * w
        # Using broadcasting: x_vals shape [BLOCK_S, BLOCK_K], w shape [BLOCK_K] -> broadcast to [BLOCK_S, BLOCK_K]
        acc0 += tl.sum(x_vals * w0[None, :], axis=1)  # [BLOCK_S]
        acc1 += tl.sum(x_vals * w1[None, :], axis=1)  # [BLOCK_S]
        acc2 += tl.sum(x_vals * w2[None, :], axis=1)  # [BLOCK_S]

    # Add biases
    b0 = tl.load(b0_ptr + h)
    b1 = tl.load(b1_ptr + h)
    b2 = tl.load(b2_ptr + h)
    acc0 += b0
    acc1 += b1
    acc2 += b2

    # Store results
    B_out_ptrs = B_out_ptr + b * B_out_stride_b + s_offsets * B_out_stride_s + h * B_out_stride_h
    C_out_ptrs = C_out_ptr + b * C_out_stride_b + s_offsets * C_out_stride_s + h * C_out_stride_h
    X_out_ptrs = X_out_ptr + b * X_out_stride_b + s_offsets * X_out_stride_s + h * X_out_stride_h
    tl.store(B_out_ptrs, acc0, mask=mask_s)
    tl.store(C_out_ptrs, acc1, mask=mask_s)
    tl.store(X_out_ptrs, acc2, mask=mask_s)


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


# 3) Grouped causal 1D convolution on Bx: conv_out[B, H, S], kernel_size=4, groups=H
#    For each output channel c: conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t + (k - 1)] * conv_weight[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,     # *f32, (B, S, H)
    convW_ptr, convB_ptr,  # *f32, (H, H, 4), (H,)
    conv_out_ptr,          # *f32, (B, H, S)
    B, S, H,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride0, convW_stride1, convW_stride2,
    conv_out_stride_b, conv_out_stride_s, conv_out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    s_block = tl.program_id(2)

    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Load conv weights for this channel c
    w0 = tl.load(convW_ptr + c * convW_stride0 + c * convW_stride1 + 0 * convW_stride2)
    w1 = tl.load(convW_ptr + c * convW_stride0 + c * convW_stride1 + 1 * convW_stride2)
    w2 = tl.load(convW_ptr + c * convW_stride0 + c * convW_stride1 + 2 * convW_stride2)
    w3 = tl.load(convW_ptr + c * convW_stride0 + c * convW_stride1 + 3 * convW_stride2)
    b_c = tl.load(convB_ptr + c)

    # Compute conv_out[b, c, s_offsets] with causal left padding: t + (k - 1)
    # This matches F.conv1d with pad=(k-1) for k in [1..4].
    # Manually handle each tap with proper masked loads.
    # k=0: t - 1
    t_minus1 = s_offsets - 1
    mask_t_minus1 = (t_minus1 >= 0) & mask_s
    bx0 = tl.load(Bx_ptr + b * Bx_stride_b + t_minus1 * Bx_stride_s + c * Bx_stride_h, mask=mask_t_minus1, other=0.0)
    acc += bx0 * w0

    # k=1: t
    mask_t = mask_s
    bx1 = tl.load(Bx_ptr + b * Bx_stride_b + s_offsets * Bx_stride_s + c * Bx_stride_h, mask=mask_t, other=0.0)
    acc += bx1 * w1

    # k=2: t + 1
    t_plus1 = s_offsets + 1
    mask_t_plus1 = (t_plus1 < S) & mask_s
    bx2 = tl.load(Bx_ptr + b * Bx_stride_b + t_plus1 * Bx_stride_s + c * Bx_stride_h, mask=mask_t_plus1, other=0.0)
    acc += bx2 * w2

    # k=3: t + 2
    t_plus2 = s_offsets + 2
    mask_t_plus2 = (t_plus2 < S) & mask_s
    bx3 = tl.load(Bx_ptr + b * Bx_stride_b + t_plus2 * Bx_stride_s + c * Bx_stride_h, mask=mask_t_plus2, other=0.0)
    acc += bx3 * w3

    acc += b_c

    conv_out_ptrs = conv_out_ptr + b * conv_out_stride_b + c * conv_out_stride_h + s_offsets * conv_out_stride_s
    tl.store(conv_out_ptrs, acc, mask=mask_s)


# 4) Output gating: y = C * conv_out, (B, H, S)
@triton.jit
def elemwise_mul_bsh_kernel(
    C_ptr, conv_out_ptr, Out_ptr,
    B, S, H,
    C_stride_b, C_stride_s, C_stride_h,
    conv_out_stride_b, conv_out_stride_s, conv_out_stride_h,
    Out_stride_b, Out_stride_s, Out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    C_ptrs = C_ptr + b * C_stride_b + s_offsets * C_stride_s + h * C_stride_h
    conv_ptrs = conv_out_ptr + b * conv_out_stride_b + s_offsets * conv_out_stride_s + h * conv_out_stride_h
    Out_ptrs = Out_ptr + b * Out_stride_b + s_offsets * Out_stride_s + h * Out_stride_h

    C_vals = tl.load(C_ptrs, mask=mask_s, other=0.0)
    conv_vals = tl.load(conv_ptrs, mask=mask_s, other=0.0)
    Out_vals = C_vals * conv_vals
    tl.store(Out_ptrs, Out_vals, mask=mask_s)


# 5) Final linear projection: y[B, S, H] -> out[B, S, H] using out_proj_weight (H,H), bias (H)
@triton.jit
def final_linear_bsh_kernel(
    y_ptr, outW_ptr, outB_ptr, out_ptr,
    B, S, H,
    y_stride_b, y_stride_s, y_stride_h,
    outW_stride0, outW_stride1,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)

    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # y[b, s_offsets, k_offsets] -> [BLOCK_S, BLOCK_K]
        y_ptrs = y_ptr + b * y_stride_b + s_offsets[:, None] * y_stride_s + k_offsets[None, :] * y_stride_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)

        # outW[h, k] -> [BLOCK_K]
        outW_ptrs = outW_ptr + h * outW_stride0 + k_offsets * outW_stride1
        outW = tl.load(outW_ptrs, mask=mask_k, other=0.0)

        acc += tl.sum(y_vals * outW[None, :], axis=1)  # [BLOCK_S]

    # add bias
    outB = tl.load(outB_ptr + h)
    acc += outB

    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)


# ModelNew: Triton-based fused implementation
class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure dtype float32 for simplicity
        device = x.device
        dtype = x.dtype

        B, S, H = x.shape
        # Kernel sizes/tile sizes
        BLOCK_S = 128
        BLOCK_K = 64

        # 1) Triple linear projection: slice in_proj_weight into three groups (H,H)
        W0 = in_proj_weight[:H, :].contiguous()  # (H, H)
        b0 = in_proj_bias[:H].contiguous()       # (H,)
        W1 = in_proj_weight[H:2*H, :].contiguous()  # (H, H)
        b1 = in_proj_bias[2*H:3*H].contiguous()     # (H,)
        W2 = in_proj_weight[2*H:3*H, :].contiguous()  # (H, H)
        b2 = in_proj_bias[3*H:].contiguous()       # (H,)

        # Allocate outputs (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=dtype)
        C_out = torch.empty((B, S, H), device=device, dtype=dtype)
        X_out = torch.empty((B, S, H), device=device, dtype=dtype)

        # Launch triple linear kernel
        grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        triple_linear_bsh_kernel[grid](
            x, W0, b0, W1, b1, W2, b2, B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating Bx = B * X
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_mul = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out[B, H, S] with kernel_size=4, groups=H
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        grid_conv = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C * conv_out -> (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_mul2 = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 5) Final linear projection to (B, S, H)
        out = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_final = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        final_linear_bsh_kernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
