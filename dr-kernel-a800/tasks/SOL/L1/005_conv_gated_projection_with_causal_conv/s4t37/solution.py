import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I,)
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
    # grid: (B*S,) one program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # base pointers for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels I
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)

        # reduce over H dimension in tiles
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            # convert to float32 for stable accumulation
            x_vals = x_vals.to(tl.float32)

            # load weight row for output channel i
            w_vals = tl.load(W_ptr + i * w_i_stride + h_offsets * w_h_stride, mask=h_mask, other=0.0)
            w_vals = w_vals.to(tl.float32)

            # accumulate dot product
            acc += tl.sum(x_vals * w_vals, axis=0)

        # add bias if present
        bias_i = tl.load(Bias_ptr + i)
        bias_i = bias_i.to(tl.float32)
        acc += bias_i

        # store result to Out[b, s, i]
        out_ptr = out_base + i * out_i_stride
        tl.store(out_ptr, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr,         # *const float, input padded Bx: (B, H, S_in)
    W_ptr,         # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,      # *const float, conv_bias: (H,)
    Out_ptr,       # *float, output conv_out: (B, H, S_out)
    B: tl.int32,
    H: tl.int32,
    S_out: tl.int32,
    S_in: tl.int32,  # S_in = S_out + K - 1, K=4 so padding=3
    # strides
    x_b_stride: tl.int32, x_h_stride: tl.int32, x_s_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # grid: (B, H), one program per (b, g)
    b = tl.program_id(axis=0)
    g = tl.program_id(axis=1)

    # base pointers
    x_base = X_ptr + b * x_b_stride + g * x_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # iterate over output positions t in tiles
    for t in range(0, S_out, BLOCK_T):
        t_offsets = t + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < S_out

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # kernel_size K=4; causal padding: t + k - 1
        for k in range(0, 4):
            idx = t_offsets + k - 1
            in_bounds = mask_t & (idx >= 0) & (idx < S_in)
            x_vals = tl.load(x_base + idx * x_s_stride, mask=in_bounds, other=0.0)
            # conv weight for group g and kernel k
            w_val = tl.load(W_ptr + g * w_g_stride + k * w_k_stride)
            w_val = w_val.to(tl.float32)
            acc += x_vals * w_val

        # add bias
        bias_g = tl.load(Bias_ptr + g).to(tl.float32)
        acc += bias_g

        # store results
        tl.store(out_base + t_offsets * out_s_stride, acc, mask=mask_t)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Bias_ptr,      # *const float, out_proj_bias: (H,)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_o_stride: tl.int32, w_i_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid: (B*S,) one program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)

        # reduce over input H dimension
        for h_in in range(0, H, BLOCK_H):
            h_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + h_out * w_o_stride + h_offsets * w_i_stride, mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)

        # add bias
        bias_h = tl.load(Bias_ptr + h_out).to(tl.float32)
        acc += bias_h

        # store
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        """
        Triton-only implementation of the original flow:
        1) in_proj: x -> BCx (B, S, 3*H)
        2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        3) Bx = B_tensor * x_proj_tensor
        4) Grouped causal 1D conv on Bx (K=4), groups=H, stride=1, padding=K-1
        5) y = C_tensor * conv_out
        6) out_proj: y -> output (B, S, H)
        """

        B, S, H = x.shape
        K = conv_weight.shape[2]
        I = 3 * H  # triple projection

        # 1) in_proj linear via Triton
        # ensure inputs are float32 for stable compute
        x_in = x.contiguous().to(torch.float32)
        w_in = in_proj_weight.contiguous().to(torch.float32)
        b_in = in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else torch.zeros(w_in.shape[0], device=x.device, dtype=torch.float32)

        BCx = torch.empty((B, S, I), dtype=torch.float32, device=x.device)

        in_proj_linear_kernel[(B * S,)](
            x_in, w_in, b_in, BCx,
            B, S, H, I,
            x_in.stride(0), x_in.stride(1), x_in.stride(2),
            w_in.stride(0), w_in.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx
        B_tensor = BCx[:, :, :H]             # (B, S, H)
        x_proj_tensor = BCx[:, :, 2 * H:]    # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]        # (B, S, H)

        # Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 3) Pad Bx for causal conv: pad left by K-1
        # Use PyTorch for pad to avoid writing another Triton kernel; this is not heavy for these sizes.
        Bx_padded = torch.nn.functional.pad(Bx, (K - 1, 0))  # (B, S, H + K - 1)
        Bx_padded = Bx_padded.contiguous().to(torch.float32)  # (B, S, S_out + K - 1) where S_out = S

        # Conv weight: (H, 1, 4), conv_bias: (H,)
        conv_w = conv_weight.contiguous().to(torch.float32)
        conv_b = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else torch.zeros(conv_w.shape[0], device=x.device, dtype=torch.float32)

        # conv_out: (B, H, S_out)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        grouped_causal_conv1d_kernel[(B, H)](
            Bx_padded, conv_w, conv_b, conv_out,
            B, H, S, S + K - 1,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_w.stride(0), conv_w.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256,
            num_warps=4,
        )

        # 4) Output gating: y = C_tensor * conv_out
        # C_tensor: (B, S, H), conv_out: (B, H, S)
        conv_out_t = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = C_tensor * conv_out_t  # (B, S, H)

        # 5) out_proj linear via Triton
        w_out = out_proj_weight.contiguous().to(torch.float32)
        b_out = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else torch.zeros(w_out.shape[1], device=x.device, dtype=torch.float32)

        output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)

        out_proj_linear_kernel[(B * S,)](
            y, w_out, b_out, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            w_out.stride(0), w_out.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
