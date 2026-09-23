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
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S*I (one program per output element)
    pid = tl.program_id(axis=0)
    total = B * S * I
    if pid >= total:
        return

    # decode (b, s, i) from pid
    b = pid // (S * I)
    rem = pid % (S * I)
    s = rem // I
    i = rem % I

    # base pointers for this (b, s, i)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_offset = b * out_b_stride + s * out_s_stride + i * out_i_stride

    # accumulate in float32
    acc = tl.zeros((), dtype=tl.float32)

    # reduce over H
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        x_vals = tl.load(
            x_base + h_offsets * x_h_stride,
            mask=h_mask,
            other=0.0
        ).to(tl.float32)

        w_vals = tl.load(
            W_ptr + i * W_ptr.stride(0) + h_offsets * W_ptr.stride(1),
            mask=h_mask,
            other=0.0
        ).to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    # add bias
    bval = tl.load(Bias_ptr + i).to(tl.float32)
    acc += bval

    # store as float32 (Out_ptr is float32 tensor in forward)
    tl.store(Out_ptr + out_offset, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *const float, input Bx (padded): (B, H, S_in)
    W_ptr,          # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const float, conv_bias: (H,)
    Out_ptr,        # *float, output conv_out: (B, H, S_out)
    B: tl.int32,
    H: tl.int32,
    S_in: tl.int32,  # input length of padded Bx
    S_out: tl.int32, # output length (S)
    # strides
    bx_b_stride: tl.int32, bx_h_stride: tl.int32, bx_s_stride: tl.int32,
    w_h_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # Grid: (B, H) one program per (b, g)
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    b = pid_b
    g = pid_h  # g = pid_h, H groups

    # Loop over output positions t in tiles
    for t0 in range(0, S_out, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S_out

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # causal conv with K=4, padding=K-1
        # acc[t] = sum_{k=0..3} Bx[b, g, t + k - 1] * W[g, k] + Bias[g]
        for k in range(4):
            idx = t_offsets + k - 1
            # valid if idx in [0, S_in)
            valid = (idx >= 0) & (idx < S_in) & t_mask
            bx_ptr = Bx_ptr + b * bx_b_stride + g * bx_h_stride + idx * bx_s_stride
            bx_val = tl.load(bx_ptr, mask=valid, other=0.0).to(tl.float32)

            w_val = tl.load(W_ptr + g * w_h_stride + k * w_k_stride).to(tl.float32)
            acc += bx_val * w_val

        # add bias
        bias_val = tl.load(Bias_ptr + g).to(tl.float32)
        acc += bias_val

        # store to Out
        out_ptr = Out_ptr + b * out_b_stride + g * out_h_stride + t_offsets * out_s_stride
        tl.store(out_ptr, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    W_ptr,          # *const float, out_proj_weight: (H, H)
    Bias_ptr,       # *const float, out_proj_bias: (H,)
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    if pid >= B * S:
        return

    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # reduce over H_in
        for h_in in range(0, H, BLOCK_H):
            h_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + h_out * W_ptr.stride(0) + h_offsets * W_ptr.stride(1), mask=h_mask, other=0.0).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        # add bias
        bval = tl.load(Bias_ptr + h_out).to(tl.float32)
        acc += bval

        # store
        tl.store(Out_ptr + b * out_b_stride + s * out_s_stride + h_out * out_h_stride, acc)


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
        """
        Triton-optimized fused implementation:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3*H)
        2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        3) Gating: Bx = B_tensor * x_proj_tensor
        4) Grouped causal conv (K=4, groups=H) on Bx (after left pad)
        5) Output gating: y = C_tensor * conv_out
        6) out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        """
        assert x.ndim == 3, "x must be (B, S, H)"
        B, S, H = x.shape
        # Ensure contiguous and float32 for kernels (typical input dtype)
        x_ = x.contiguous().to(torch.float32)

        # 1) in_proj linear: (B, S, I) where I = 3*H
        I = 3 * H
        W_in = in_proj_weight.contiguous().to(torch.float32)  # (I, H)
        bias_in = in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else torch.zeros(I, device=x_.device, dtype=torch.float32)

        BCx = torch.empty((B, S, I), dtype=torch.float32, device=x_.device)

        grid_in = (B * S * I,)
        in_proj_linear_kernel[grid_in](
            x_, W_in, bias_in, BCx,
            B, S, H, I,
            x_.stride(0), x_.stride(1), x_.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        BCx_c = BCx  # (B, S, I)
        B_tensor = BCx_c[:, :, :H]           # (B, S, H)
        x_proj_tensor = BCx_c[:, :, 2 * H:]  # (B, S, H)
        C_tensor = BCx_c[:, :, H:2 * H]      # (B, S, H)

        # Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 3) Grouped causal conv: with K=4, padding=K-1=3, groups=H
        # Left pad Bx along sequence dimension
        pad = 3  # K-1
        S_in = S + pad
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # pad left by 3, right by 0 → (B, H, S+3)
        Bx_padded = Bx_padded.contiguous().to(torch.float32)  # (B, H, S+3)

        conv_w = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4)
        conv_b = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else torch.zeros(conv_w.shape[0], device=x_.device, dtype=torch.float32)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x_.device)

        grid_conv = (B, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_w, conv_b, conv_out,
            B, H, S_in, S,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_w.stride(0), conv_w.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256,
            num_warps=4,
        )

        # 4) Output gating: y = C_tensor * conv_out
        # conv_out: (B, H, S), C_tensor: (B, S, H)
        conv_out_t = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = C_tensor * conv_out_t  # (B, S, H)

        # 5) out_proj linear: (B, S, H) -> (B, S, H)
        y_ = y.contiguous().to(torch.float32)
        W_out = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        Bias_out = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else torch.zeros(W_out.shape[1], device=x_.device, dtype=torch.float32)

        output = torch.empty((B, S, H), dtype=torch.float32, device=x_.device)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y_, W_out, Bias_out, output,
            B, S, H,
            y_.stride(0), y_.stride(1), y_.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
