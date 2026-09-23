import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    X_ptr,              # *const float, input x: (B, S, H)
    W_ptr,              # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,            # *float, output BCx: (B, S, I)
    B: tl.int32,        # batch size
    S: tl.int32,        # sequence length
    H: tl.int32,        # hidden size
    I: tl.int32,        # output channels = 3*H
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # base pointers for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels I (each corresponds to a linear combination over H)
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)

        # reduce over H dimension
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            # load weight row for output channel i
            w_vals = tl.load(
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(x_vals * w_vals, axis=0)

        # store result to Out[b, s, i]
        out_ptr_i = out_base + i * out_i_stride
        tl.store(out_ptr_i, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,             # *const float, input for conv: (B, H, S) from transposed x after in_proj gating
    W_ptr,              # *const float, conv_weight: (H, 1, 4), groups=H
    Bias_ptr,           # *const float, conv_bias: (H,)
    Out_ptr,            # *float, output conv_out: (B, H, S)
    B: tl.int32,        # batch size
    S: tl.int32,        # sequence length
    H: tl.int32,        # hidden size
    K: tl.constexpr,    # kernel_size (fixed 4)
    BLOCK_T: tl.constexpr,  # tile along S dimension for output
):
    # Grid is (B * H,). Each program handles one (b, g) pair.
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # base pointer for this (b, g)
    Bx_base = Bx_ptr + b * (H * S) + g * S  # (B, H, S) layout: (b,h) plane offset plus s index

    # Output base
    Out_base = Out_ptr + b * (H * S) + g * S

    # Process S in tiles
    for t0 in range(0, S, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # Convolution over K=4 with causal padding: t + k - 1
        for k in range(0, K):
            pos = t_offsets + k - 1  # causal: we use input at pos=t-k+1
            valid = (pos >= 0) & (pos < S) & t_mask

            # Load Bx[b, g, pos] = x[b, pos, g] (transposed view)
            bx_ptrs = Bx_base + pos
            bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0).to(tl.float32)

            # Load conv weight for group g and kernel k: conv_weight[g, 0, k]
            w_val = tl.load(W_ptr + g * (1 * K) + 0 * K + k).to(tl.float32)

            acc += bx_vals * w_val

        # Add bias
        bias_val = tl.load(Bias_ptr + g).to(tl.float32)
        acc += bias_val

        # Store to Out[b, g, t_offsets]
        out_ptrs = Out_base + t_offsets
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_kernel(
    Y_ptr,              # *const float, input y: (B, S, H)
    W_ptr,              # *const float, out_proj_weight: (H, H)
    Out_ptr,            # *float, output: (B, S, H)
    B: tl.int32,        # batch size
    S: tl.int32,        # sequence length
    H: tl.int32,        # hidden size
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_out_stride: tl.int32, w_in_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H reduction
    BLOCK_T: tl.constexpr,  # tile size along output H (we produce H)
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H, BLOCK_H):
        h_out_offsets = h_out + tl.arange(0, BLOCK_H)
        h_out_mask = h_out_offsets < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Reduction over input H
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            # y[b, s, h_in]
            y_ptrs = y_base + h_in_offsets * y_h_stride
            y_vals = tl.load(
                y_ptrs,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            # W[h_out, h_in] as 2D tile
            w_ptrs = W_ptr + h_out_offsets[:, None] * w_out_stride + h_in_offsets[None, :] * w_in_stride
            w_mask = (h_out_mask[:, None]) & (h_in_mask[None, :])
            w_vals = tl.load(
                w_ptrs,
                mask=w_mask,
                other=0.0
            ).to(tl.float32)

            # acc[h_out] += sum over h_in of y[b,s,h_in] * W[h_out, h_in]
            acc += tl.sum(y_vals[None, :] * w_vals, axis=1)

        # Store results
        out_ptrs = out_base + h_out_offsets * out_h_stride
        tl.store(out_ptrs, acc, mask=h_out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        1) Triton in_proj: x -> (B, S, 3*H)
        2) PyTorch slicing and elementwise gating: B = first H, x_proj = third H, Bx = B * x_proj
        3) Triton grouped causal conv with kernel_size=4, groups=H, output (B, H, S)
        4) PyTorch gating: y = C * conv_out (C is second H from in_proj)
        5) Triton out_proj: y -> (B, S, H)
        """
        # Shapes
        B, S, H = x.shape
        I = 3 * H

        device = x.device
        dtype = torch.float32  # keep compute in fp32

        # Ensure contiguity and dtype for kernels
        x_contig = x.contiguous().to(dtype)
        in_proj_weight_contig = in_proj_weight.contiguous().to(dtype)
        conv_weight_contig = conv_weight.contiguous().to(dtype)
        conv_bias_contig = conv_bias.contiguous().to(dtype)
        out_proj_weight_contig = out_proj_weight.contiguous().to(dtype)

        # 1) Triton in_proj: BCx = X @ W_in^T, shape (B, S, I)
        BCx = torch.empty((B, S, I), device=device, dtype=dtype)
        grid_in = (B * S,)
        in_proj_kernel[grid_in](
            x_contig, in_proj_weight_contig, BCx,
            B, S, H, I,
            x_b_stride=x_contig.stride(0), x_s_stride=x_contig.stride(1), x_h_stride=x_contig.stride(2),
            w_i_stride=in_proj_weight_contig.stride(0), w_h_stride=in_proj_weight_contig.stride(1),
            out_b_stride=BCx.stride(0), out_s_stride=BCx.stride(1), out_i_stride=BCx.stride(2),
            BLOCK_H=128,
            num_warps=4,
        )

        # 2) Slicing and gating in PyTorch (view ops)
        # BCx shape (B,S,I), I=3H -> split into B, C, x_proj along last dim
        B_tensor = BCx[:, :, :H]        # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]     # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:] # (B, S, H)
        Bx = B_tensor * x_proj_tensor   # elementwise gating

        # 3) Triton grouped causal conv: input Bx transposed to (B, H, S)
        Bx_trans = Bx.transpose(-1, -2).contiguous()  # (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight_contig, conv_bias_contig, conv_out,
            B, S, H,
            K=4,
            BLOCK_T=128,
            num_warps=4,
        )

        # 4) PyTorch gating: y = C * conv_out
        # C_tensor: (B, S, H), conv_out: (B, H, S)
        y = C_tensor * conv_out.transpose(-1, -2)  # (B, S, H) elementwise multiply

        # 5) Triton out_proj: y -> output (B, S, H)
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_out = (B * S,)
        out_proj_kernel[grid_out](
            y, out_proj_weight_contig, output,
            B, S, H,
            y_b_stride=y.stride(0), y_s_stride=y.stride(1), y_h_stride=y.stride(2),
            w_out_stride=out_proj_weight_contig.stride(0), w_in_stride=out_proj_weight_contig.stride(1),
            out_b_stride=output.stride(0), out_s_stride=output.stride(1), out_h_stride=output.stride(2),
            BLOCK_H=128,
            BLOCK_T=128,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
