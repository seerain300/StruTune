import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32, S: tl.int32, H: tl.int32, I: tl.int32,
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # For each output channel i in [0, I)
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + i * w_i_stride + h_offsets * w_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            # FMA over vector
            acc += tl.sum(x_vals * w_vals, axis=0)
        # Store scalar to out[b, s, i]
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def pad_left_kernel(
    In_ptr,        # *const float, input X (B, H, T)
    Out_ptr,       # *float, output with left pad (B, H, T+pad)
    B: tl.int32, H: tl.int32, T: tl.int32, pad: tl.int32,
    in_b_stride: tl.int32, in_h_stride: tl.int32, in_t_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # Grid: axis=0 over B*H (one program per (b, h))
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    in_base = In_ptr + b * in_b_stride + h * in_h_stride
    out_base = Out_ptr + b * out_b_stride + h * out_h_stride

    for t0 in range(0, T, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T
        vals = tl.load(in_base + t_offsets * in_t_stride, mask=t_mask, other=0.0).to(tl.float32)
        # store shifted by pad: out[t+pad] = in[t]
        out_t_offsets = t_offsets + pad
        out_mask = (t_offsets < T) & (out_t_offsets < (T + pad))
        tl.store(out_base + out_t_offsets * out_t_stride, vals, mask=out_mask)


@triton.jit
def grouped_causal_conv1d_kernel(
    X_pad_ptr,      # *const float, input after left pad: (B, H, T+pad), T=S+pad-1 but here pad=3
    W_ptr,          # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const float, conv_bias: (H)
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32, H: tl.int32, S: tl.int32, pad: tl.int32,  # pad=3
    x_b_stride: tl.int32, x_h_stride: tl.int32, x_t_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # Grid: axis=0 over B*H (one program per (b, g))
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # base pointers for this (b, g)
    x_base = X_pad_ptr + b * x_b_stride + g * x_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # We compute conv_out[b, g, :] for all t in [0..S-1]
    for t0 in range(0, S, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Kernel size K=4, causal: t_in = t + k - pad
        # k=0
        t_in0 = t_offsets + 0 - pad
        valid0 = t_mask & (t_in0 >= 0)
        x0 = tl.load(x_base + t_in0 * x_t_stride, mask=valid0, other=0.0)
        w0 = tl.load(W_ptr + g * w_g_stride + 0 * w_k_stride, mask=True, other=0.0)
        acc += x0 * w0

        # k=1
        t_in1 = t_offsets + 1 - pad
        valid1 = t_mask & (t_in1 >= 0)
        x1 = tl.load(x_base + t_in1 * x_t_stride, mask=valid1, other=0.0)
        w1 = tl.load(W_ptr + g * w_g_stride + 1 * w_k_stride, mask=True, other=0.0)
        acc += x1 * w1

        # k=2
        t_in2 = t_offsets + 2 - pad
        valid2 = t_mask & (t_in2 >= 0)
        x2 = tl.load(x_base + t_in2 * x_t_stride, mask=valid2, other=0.0)
        w2 = tl.load(W_ptr + g * w_g_stride + 2 * w_k_stride, mask=True, other=0.0)
        acc += x2 * w2

        # k=3
        t_in3 = t_offsets + 3 - pad
        valid3 = t_mask & (t_in3 >= 0)
        x3 = tl.load(x_base + t_in3 * x_t_stride, mask=valid3, other=0.0)
        w3 = tl.load(W_ptr + g * w_g_stride + 3 * w_k_stride, mask=True, other=0.0)
        acc += x3 * w3

        # add bias for group g
        bias = tl.load(Bias_ptr + g, mask=True, other=0.0)
        acc += bias

        # store to Out[b, g, t]
        tl.store(out_base + t_offsets * out_s_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    W_ptr,          # *const float, out_proj_weight: (H, H)
    Bias_ptr,       # *const float, out_proj_bias: (H) or None
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_out_h_stride: tl.int32, w_in_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for ho in range(0, H, BLOCK_H):
        h_out_offsets = ho + tl.arange(0, BLOCK_H)
        h_out_mask = h_out_offsets < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        # reduce over input H
        for hi in range(0, H, BLOCK_H):
            h_in_offsets = hi + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=h_in_mask, other=0.0).to(tl.float32)  # (BLOCK_H)
            w_vals = tl.load(W_ptr + h_out_offsets[:, None] * w_out_h_stride + h_in_offsets[None, :] * w_in_h_stride,
                             mask=h_out_mask[:, None] & h_in_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_H, BLOCK_H)
            # acc[h_out] += sum_h (y[b, s, h_in] * W[h_out, h_in])
            acc += tl.sum(w_vals * y_vals[None, :], axis=1)

        # add bias
        bias_vals = tl.load(Bias_ptr + h_out_offsets, mask=h_out_mask, other=0.0).to(tl.float32)
        acc += bias_vals

        # store
        tl.store(out_base + h_out_offsets * out_h_stride, acc, mask=h_out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        B, S, H = x.shape
        I = 3 * H  # in_proj output channels

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # Allocate BCx in the same dtype as x for correctness; compute in Triton in float32 and store in that dtype
        bcx = torch.empty((B, S, I), device=x.device, dtype=x.dtype)
        x_contig = x.contiguous()
        w_contig = in_proj_weight.contiguous()
        # Strides (assuming contiguous)
        x_b_stride = x_contig.stride(0)
        x_s_stride = x_contig.stride(1)
        x_h_stride = x_contig.stride(2)
        w_i_stride = w_contig.stride(0)  # rows
        w_h_stride = w_contig.stride(1)  # cols
        out_b_stride = bcx.stride(0)
        out_s_stride = bcx.stride(1)
        out_i_stride = bcx.stride(2)

        BLOCK_H = 64
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_contig, w_contig, bcx,
            B, S, H, I,
            x_b_stride, x_s_stride, x_h_stride,
            w_i_stride, w_h_stride,
            out_b_stride, out_s_stride, out_i_stride,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # 2) Slice BCx into B_tensor, C_tensor, x_proj_tensor
        B_tensor = bcx[:, :, :H]
        C_tensor = bcx[:, :, H:2*H]
        x_proj_tensor = bcx[:, :, 2*H:]

        # Elementwise gating: Bx = B_tensor * x_proj_tensor (PyTorch; not heavy)
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 3) Pad for causal conv: left pad by pad = conv_kernel_size - 1 = 3
        Bx_pad = torch.empty((B, H, S + 3), device=x.device, dtype=x.dtype)
        # Use Triton pad kernel
        in_b_stride = Bx.stride(0)
        in_h_stride = Bx.stride(1)
        in_t_stride = Bx.stride(2)
        out_b_stride = Bx_pad.stride(0)
        out_h_stride = Bx_pad.stride(1)
        out_t_stride = Bx_pad.stride(2)

        BLOCK_T = 256
        grid_pad = (B * H,)
        pad_left_kernel[grid_pad](
            Bx, Bx_pad,
            B, H, S, 3,
            in_b_stride, in_h_stride, in_t_stride,
            out_b_stride, out_h_stride, out_t_stride,
            BLOCK_T=BLOCK_T,
            num_warps=4,
        )

        # 4) Grouped causal conv: conv_out = F.conv1d(Bx_pad, conv_weight, conv_bias, groups=H)
        # conv_weight shape: (H, 1, 4), conv_bias: (H)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        w_g_stride = conv_weight.stride(0)  # H
        w_k_stride = conv_weight.stride(2)  # 4 (since w[:, :, 0] is a single element per g)
        bias = conv_bias
        # Strides for conv_in (Bx_pad)
        x_b_stride = Bx_pad.stride(0)
        x_h_stride = Bx_pad.stride(1)
        x_t_stride = Bx_pad.stride(2)
        # Strides for conv_out
        out_b_stride = conv_out.stride(0)
        out_h_stride = conv_out.stride(1)
        out_s_stride = conv_out.stride(2)

        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_pad, conv_weight, bias, conv_out,
            B, H, S, 3,
            x_b_stride, x_h_stride, x_t_stride,
            w_g_stride, w_k_stride,
            out_b_stride, out_h_stride, out_s_stride,
            BLOCK_T=BLOCK_T,
            num_warps=4,
        )

        # 5) Output gating: y = C_tensor * conv_out
        # Note: C_tensor is the original slice, not B_tensor.
        y = C_tensor * conv_out  # (B, S, H)
        # y has same dtype as x (from step 1 output), conv_out and C_tensor have same dtype as x

        # 6) Final out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        y_contig = y.contiguous()
        w_out = out_proj_weight.contiguous()
        b_bias = out_proj_bias.contiguous() if out_proj_bias is not None else torch.zeros(H, device=x.device, dtype=x.dtype)

        y_b_stride = y_contig.stride(0)
        y_s_stride = y_contig.stride(1)
        y_h_stride = y_contig.stride(2)

        w_out_h_stride = w_out.stride(0)
        w_in_h_stride = w_out.stride(1)

        out_b_stride = output.stride(0)
        out_s_stride = output.stride(1)
        out_h_stride = output.stride(2)

        BLOCK_H = 64
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y_contig, w_out, b_bias, output,
            B, S, H,
            y_b_stride, y_s_stride, y_h_stride,
            w_out_h_stride, w_in_h_stride,
            out_b_stride, out_s_stride, out_h_stride,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
