import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # Loop over output channels I
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over H
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(x_vals * w_vals, axis=0)

        # Store to Out[b, s, i]
        tl.store(out_base + i * out_i_stride, acc.to(Out_ptr.dtype.element_ty))


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *const float, input after gating: (B, H, S)
    W_ptr,          # *const float, conv weight: (H, 1, K), K=4
    Bias_ptr,       # *const float, conv bias: (H)
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    K: tl.int32,    # kernel size, here 4
    # strides
    bx_b_stride: tl.int32, bx_g_stride: tl.int32, bx_s_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_S: tl.constexpr,
):
    # 2D grid: axis=0 over B*H, axis=1 over tiles of S
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    b = pid0 // H
    g = pid0 % H

    # tile of output positions
    t_start = pid1 * BLOCK_S
    t_offsets = t_start + tl.arange(0, BLOCK_S)
    t_mask = t_offsets < S

    # accumulator for this (b, g, t tile)
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # causal padding: pad left by K-1
    for k in range(0, K):
        in_t = t_offsets + k - (K - 1)  # +k -3 for K=4
        in_mask = (in_t >= 0) & (in_t < S) & t_mask
        # load Bx[b, g, in_t]
        bx_ptrs = Bx_ptr + b * bx_b_stride + g * bx_g_stride + in_t * bx_s_stride
        bx_vals = tl.load(bx_ptrs, mask=in_mask, other=0.0).to(tl.float32)

        # load conv weight for group g, tap k
        w_val = tl.load(W_ptr + g * w_g_stride + k * w_k_stride).to(tl.float32)

        # multiply and accumulate
        acc += bx_vals * w_val

    # add bias if provided
    bias_val = tl.load(Bias_ptr + g).to(tl.float32)
    acc += bias_val

    # store result to Out[b, g, t_offsets]
    out_ptrs = Out_ptr + b * out_b_stride + g * out_g_stride + t_offsets * out_s_stride
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    Wout_ptr,       # *const float, out_proj_weight: (H, H)
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    wout_o_stride: tl.int32, wout_i_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # For each output channel h_out, compute dot over H
    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # reduce over input H
        for h_in in range(0, H, BLOCK_H):
            h_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            y_vals = tl.load(
                y_base + h_offsets * y_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                Wout_ptr + h_out * wout_o_stride + h_offsets * wout_i_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        # Store output[b, s, h_out]
        tl.store(out_base + h_out * out_h_stride, acc.to(Out_ptr.dtype.element_ty))


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure inputs are on CUDA for Triton
        assert x.is_cuda, "Input x must be on CUDA for Triton kernels."

        B, S, H = x.shape
        I = 3 * H  # in_proj outputs 3*H channels

        # 1) in_proj linear: BCx ∈ (B, S, I)
        # Make sure inputs are contiguous
        x_ = x.contiguous()
        in_proj_weight_ = in_proj_weight.contiguous()
        # Output tensor with same dtype as input
        BCx = torch.empty((B, S, I), device=x.device, dtype=x.dtype)
        in_proj_linear_kernel[(B * S,)](
            x_, in_proj_weight_, BCx,
            B, S, H, I,
            x_.stride(0), x_.stride(1), x_.stride(2),
            in_proj_weight_.stride(0), in_proj_weight_.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4, num_stages=2
        )

        # Slicing for original ops:
        B_tensor = BCx[:, :, :H]              # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]        # (B, S, H)
        x_proj_tensor = BCx[:, :, 2 * H:]    # (B, S, H)

        # Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # Transpose for conv: (B, H, S)
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, S, H) -> (B, H, S)

        # 2) Grouped causal 1D conv with kernel_size=4, groups=H
        K = 4
        conv_weight_ = conv_weight.contiguous()  # (H, 1, 4)
        conv_bias_ = conv_bias.contiguous()      # (H)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        grid = (B * H, triton.cdiv(S, 128))
        grouped_causal_conv1d_kernel[grid](
            Bx_trans, conv_weight_, conv_bias_,
            conv_out,
            B, H, S, K,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_weight_.stride(0), conv_weight_.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 3) Output gating: y = C * conv_out
        # Shapes: C_tensor: (B, H, S), conv_out: (B, H, S)
        y = C_tensor * conv_out  # (B, H, S)

        # Transpose back for out_proj: (B, S, H)
        y = y.transpose(1, 2).contiguous()  # (B, S, H)

        # 4) out_proj linear: output ∈ (B, S, H)
        out_proj_weight_ = out_proj_weight.contiguous()  # (H, H)
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        out_proj_linear_kernel[(B * S,)](
            y, out_proj_weight_, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight_.stride(0), out_proj_weight_.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
