import torch
import triton
import triton.language as tl

# 1) Triple linear projection: x[B, S, H] -> B[B,S,H], C[B,S,H], X[B,S,H]
#    where W0, W1, W2 are slices of in_proj_weight: (3*H, H) => each (H, H)
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    W0_ptr, b0_ptr,   # *f32, (H,H), (H,)
    W1_ptr, b1_ptr,   # *f32, (H,H), (H,)
    W2_ptr, b2_ptr,   # *f32, (H,H), (H,)
    out0_ptr, out1_ptr, out2_ptr,  # *f32, (B,S,H)
    B, S, H,
    x_stride_b, x_stride_s, x_stride_h,
    W0_stride0, W0_stride1,
    W1_stride0, W1_stride1,
    W2_stride0, W2_stride1,
    out0_stride_b, out0_stride_s, out0_stride_h,
    out1_stride_b, out1_stride_s, out1_stride_h,
    out2_stride_b, out2_stride_s, out2_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulators for the current h over K dimension
    acc0 = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load x[b, s, k]
        x_ptrs = x_ptr + b * x_stride_b + s_offsets[:, None] * x_stride_s + k_offsets[None, :] * x_stride_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_S, BLOCK_K]

        # Load weights W[h, k]
        w0_ptrs = W0_ptr + h * W0_stride0 + k_offsets * W0_stride1
        w1_ptrs = W1_ptr + h * W1_stride0 + k_offsets * W1_stride1
        w2_ptrs = W2_ptr + h * W2_stride0 + k_offsets * W2_stride1
        w0 = tl.load(w0_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        w1 = tl.load(w1_ptrs, mask=mask_k, other=0.0)
        w2 = tl.load(w2_ptrs, mask=mask_k, other=0.0)

        # Accumulate dot products for each k in the chunk
        # acc[i] += sum_k x_vals[i, k] * weight[k]
        for kk in range(BLOCK_K):
            k_active = k_start + kk
            k_mask = k_active < H
            # if k_mask is False, skip by masking
            # contribution = x_vals[:, kk] * (w0[kk] if k_mask else 0)
            contrib0 = tl.where(k_mask, x_vals[:, kk] * w0[kk], 0.0)
            contrib1 = tl.where(k_mask, x_vals[:, kk] * w1[kk], 0.0)
            contrib2 = tl.where(k_mask, x_vals[:, kk] * w2[kk], 0.0)
            acc0 += contrib0
            acc1 += contrib1
            acc2 += contrib2

    # Add bias
    b0 = tl.load(b0_ptr + h)
    b1 = tl.load(b1_ptr + h)
    b2 = tl.load(b2_ptr + h)
    acc0 += b0
    acc1 += b1
    acc2 += b2

    # Store results
    out0_ptrs = out0_ptr + b * out0_stride_b + s_offsets * out0_stride_s + h * out0_stride_h
    out1_ptrs = out1_ptr + b * out1_stride_b + s_offsets * out1_stride_s + h * out1_stride_h
    out2_ptrs = out2_ptr + b * out2_stride_b + s_offsets * out2_stride_s + h * out2_stride_h
    tl.store(out0_ptrs, acc0, mask=mask_s)
    tl.store(out1_ptrs, acc1, mask=mask_s)
    tl.store(out2_ptrs, acc2, mask=mask_s)

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

# 3) Grouped causal 1D convolution on Bx (B,S,H) with kernel_size=4, groups=H
#    Input Bx is (B, S, H), conv_weight is (H, H, 4), conv_bias (H)
#    Output conv_out is (B, H, S)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convB_ptr, out_ptr,
    B, S, H,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride_cout, convW_stride_cin, convW_stride_k,
    out_stride_b, out_stride_h, out_stride_s,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    c_out = tl.program_id(1)  # equals c_in because groups=H and mapping is 1:1
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for this (b, c_out) across S
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # k loop over kernel_size=4 (k=0..3)
    # Note: causal padding: Bx[b, c_out, t - k], valid when t >= k
    for k in range(4):
        t_offsets = s_offsets + k
        valid_mask = t_offsets < S
        Bx_ptrs = Bx_ptr + b * Bx_stride_b + t_offsets * Bx_stride_s + c_out * Bx_stride_h
        Bx_vals = tl.load(Bx_ptrs, mask=valid_mask, other=0.0)  # [BLOCK_S]
        convW_ptr_k = convW_ptr + c_out * convW_stride_cout + c_out * convW_stride_cin + k * convW_stride_k
        convW_k = tl.load(convW_ptr_k)  # scalar
        acc += Bx_vals * convW_k

    # Add bias
    convB_k = tl.load(convB_ptr + c_out)
    acc += convB_k

    # Store to conv_out[b, c_out, s]
    out_ptrs = out_ptr + b * out_stride_b + c_out * out_stride_h + s_offsets * out_stride_s
    tl.store(out_ptrs, acc, mask=mask_s)

# 4) Output gating: y = C * conv_out, both (B, H, S)
@triton.jit
def elemwise_mul_bsh_kernel(
    C_ptr, convOut_ptr, Out_ptr,
    B, S, H,
    C_stride_b, C_stride_s, C_stride_h,
    convOut_stride_b, convOut_stride_s, convOut_stride_h,
    Out_stride_b, Out_stride_s, Out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S
    C_ptrs = C_ptr + b * C_stride_b + s_offsets * C_stride_s + h * C_stride_h
    convOut_ptrs = convOut_ptr + b * convOut_stride_b + s_offsets * convOut_stride_s + h * convOut_stride_h
    Out_ptrs = Out_ptr + b * Out_stride_b + s_offsets * Out_stride_s + h * Out_stride_h
    C_vals = tl.load(C_ptrs, mask=mask_s, other=0.0)
    convOut_vals = tl.load(convOut_ptrs, mask=mask_s, other=0.0)
    Out_vals = C_vals * convOut_vals
    tl.store(Out_ptrs, Out_vals, mask=mask_s)

# 5) Final linear projection: y[B,S,H] -> out[B,S,H] using out_proj_weight (H,H), bias (H)
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
        y_ptrs = y_ptr + b * y_stride_b + s_offsets[:, None] * y_stride_s + k_offsets[None, :] * y_stride_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)
        outW_ptrs = outW_ptr + h * outW_stride0 + k_offsets * outW_stride1
        outW = tl.load(outW_ptrs, mask=mask_k, other=0.0)
        for kk in range(BLOCK_K):
            k_active = k_start + kk
            k_mask = k_active < H
            contrib = tl.where(k_mask, y_vals[:, kk] * outW[kk], 0.0)
            acc += contrib

    outB = tl.load(outB_ptr + h)
    acc += outB

    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)

class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # Ensure dtype and device
        device = x.device
        B, S, H = x.shape

        # 1) Triple linear projection
        # Slice in_proj_weight into three groups
        W0 = in_proj_weight[:H, :]         # (H, H)
        b0 = in_proj_bias[:H]              # (H,)
        W1 = in_proj_weight[H:2*H, :]      # (H, H)
        b1 = in_proj_bias[2*H:3*H]         # (H,)
        W2 = in_proj_weight[2*H:3*H, :]    # (H, H)
        b2 = in_proj_bias[3*H:4*H]         # (H,)

        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Grid: (B, H, ceil_div(S, BLOCK_S))
        BLOCK_S = 128
        grid_triple = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        triple_linear_bsh_kernel[grid_triple](
            x, W0, b0, W1, b1, W2, b2, B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out[b, c, s]
        convW = conv_weight.contiguous()   # (H, H, 4)
        convB = conv_bias.contiguous()     # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # (B, H, S)
        grid_conv = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 5) Final linear projection to output (B, S, H)
        out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        # out_proj_weight is (H, H), out_proj_bias (H)
        final_linear_bsh_kernel[(B, H, (S + BLOCK_S - 1) // BLOCK_S)](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
