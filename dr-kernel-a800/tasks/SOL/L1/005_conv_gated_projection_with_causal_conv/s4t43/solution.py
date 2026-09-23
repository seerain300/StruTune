import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I) or None (set to zeros in host if None)
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels I (each corresponds to a linear combination over H)
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # reduce over H dimension in tiles
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

        if Bias_ptr != 0:  # allow optional bias
            bias_val = tl.load(Bias_ptr + i).to(tl.float32)
            acc += bias_val

        # store to Out[b, s, i], cast to output dtype implicitly via pointer type
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *const float, input Bx: (B, H, S)
    W_ptr,          # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const float, conv_bias: (H)
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    bx_b_stride: tl.int32, bx_g_stride: tl.int32, bx_t_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,  # conv weight strides for (H, 1, 4)
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,  # tile along time (S)
):
    # Grid: axis=0 over B*S tiles, axis=1 over groups (H)
    pid0 = tl.program_id(axis=0)
    groups = tl.program_id(axis=1)

    # map pid0 to (b, time-tile start)
    b = pid0 // (S // BLOCK_T)
    t_start = (pid0 % (S // BLOCK_T)) * BLOCK_T

    # loop over output positions in this tile
    for t in range(0, BLOCK_T):
        t_idx = t_start + t
        if t_idx >= S:
            break

        # accumulator for this (b, group, t)
        acc = tl.zeros((), dtype=tl.float32)

        # compute contributions for k in [0..3]
        # causal padding handled by masked loads: for out-of-range t+k-1, load 0
        for k in range(0, 4):
            idx = t_idx + k - 1  # causal shifted index
            # mask for valid input
            valid = (idx >= 0) & (idx < S)
            # load from Bx[b, groups, idx] or 0 if invalid
            bx_val = tl.load(
                Bx_ptr + b * bx_b_stride + groups * bx_g_stride + idx * bx_t_stride,
                mask=valid,
                other=0.0
            ).to(tl.float32)

            w_val = tl.load(
                W_ptr + groups * w_g_stride + k * w_k_stride
            ).to(tl.float32)

            acc += bx_val * w_val

        # add bias for this group
        if Bias_ptr != 0:
            bias_val = tl.load(Bias_ptr + groups).to(tl.float32)
            acc += bias_val

        # store result
        tl.store(
            Out_ptr + b * out_b_stride + groups * out_g_stride + t_idx * out_t_stride,
            acc
        )


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    W_ptr,          # *const float, out_proj_weight: (H, H)
    Bias_ptr,       # *const float, out_proj_bias: (H) or None
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_hout_stride: tl.int32, w_hin_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                W_ptr + h_out * w_hout_stride + h_in_offsets * w_hin_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        if Bias_ptr != 0:
            bias_val = tl.load(Bias_ptr + h_out).to(tl.float32)
            acc += bias_val

        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized forward that performs:
        1) in_proj: x -> BCx = F.linear(x, in_proj_weight, in_proj_bias) with I=3*H
        2) Slice BCx into B_tensor, C_tensor, x_proj_tensor
        3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        4) Grouped causal 1D conv on Bx (no torch.conv): conv_out
        5) y = C_tensor * conv_out
        6) out_proj: y -> output
        All heavy ops (in_proj, conv, out_proj) are Triton kernels.
        """
        B, S, H = x.shape
        I = 3 * H  # in_proj output channels

        # 1) in_proj linear: BCx = x @ in_proj_weight^T + in_proj_bias
        # Make sure inputs are contiguous
        x_contig = x.contiguous()
        w_in = in_proj_weight.contiguous()
        # allocate output BCx with same dtype/device as x
        BCx = torch.empty((B, S, I), dtype=x_contig.dtype, device=x_contig.device)

        # Launch in_proj_kernel
        BLOCK_H = 64
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_contig, w_in, in_proj_bias if in_proj_bias is not None else x_contig, BCx,
            B, S, H, I,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            w_in.stride(0), w_in.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # 2) Elementwise gating
        # Split BCx into B_tensor, C_tensor, x_proj_tensor
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2*H]
        x_proj_tensor = BCx[:, :, 2*H:]

        # Bx = B_tensor * x_proj_tensor  (PyTorch elementwise multiply; not heavy)
        Bx = B_tensor * x_proj_tensor  # shape (B, H, S)

        # For conv, we need Bx as (B, H, S). We'll operate directly on Bx.

        # 3) Grouped causal conv1d: conv_out = conv(Bx, conv_weight, groups=H)
        # conv_weight: (H, 1, 4), conv_bias: (H)
        # Allocate conv_out
        conv_out = torch.empty((B, H, S), dtype=Bx.dtype, device=Bx.device)

        # Launch grouped_causal_conv1d_kernel
        # Grid: axis=0 over B*S tiles, axis=1 over groups H
        BLOCK_T = 256
        grid_conv = ( (B * S + BLOCK_T - 1) // BLOCK_T, H )
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_weight.contiguous(), conv_bias if conv_bias is not None else conv_out, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),  # w_k_stride corresponds to kernel dim (4)
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=BLOCK_T,
            num_warps=4,
        )

        # 4) Output gating with C_tensor
        # y = C_tensor.transpose(1, 2) * conv_out
        # C_tensor is (B, H, S), conv_out is (B, H, S)
        C_t = C_tensor.transpose(1, 2)  # (B, S, H)
        y = C_t * conv_out  # elementwise multiply

        # 5) out_proj linear: output = y @ out_proj_weight^T + out_proj_bias
        # y: (B, S, H), out_proj_weight: (H, H)
        output = torch.empty((B, S, H), dtype=y.dtype, device=y.device)

        w_out = out_proj_weight.contiguous()
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, w_out, out_proj_bias if out_proj_bias is not None else output, output,
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
