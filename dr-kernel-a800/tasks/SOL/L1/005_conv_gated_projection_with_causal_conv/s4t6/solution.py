import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float32, input x: (B, S, H)
    W_ptr,         # *const float32, in_proj_weight: (I, H), I=3*H
    Out_ptr,       # *float32, output BCx: (B, S, I)
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
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride

    # Loop over output channels I
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)

        # Reduce over input hidden dimension H
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            w_vals = tl.load(W_ptr + i * H + h_offsets, mask=h_mask, other=0.0)

            acc += tl.sum(x_vals * w_vals, axis=0)

        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + i * out_i_stride
        tl.store(out_ptr, acc)  # store as float32


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float32, input Bx: (B, H, S), transposed from (B, S, H)
    W_ptr,         # *const float32, conv_weight: (H, 1, 4)
    Bias_ptr,      # *const float32, conv_bias: (H)
    Out_ptr,       # *float32, output conv_out: (B, H, S)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    K: tl.constexpr,        # kernel size, here 4
    BLOCK_T: tl.constexpr,  # tile along S
):
    # Grid over (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    bx_base = Bx_ptr + b * (H * S) + g * S
    out_base = Out_ptr + b * (H * S) + g * S

    # Loop over output positions t in tiles
    for t0 in range(0, S, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # Accumulate over K=4 with causal padding: input index = t + k - 1
        for k in range(0, K):
            pos = t_offsets + k - 1
            valid = (pos >= 0) & (pos < S) & t_mask
            bx_ptrs = bx_base + pos
            bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0)
            acc += bx_vals

        # Add bias[g]
        bias_val = tl.load(Bias_ptr + g)
        acc += bias_val

        out_ptrs = out_base + t_offsets
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float32, input y: (B, S, H)
    W_ptr,         # *const float32, out_proj_weight: (H, H)
    Out_ptr,       # *float32, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    # strides for W
    w_out_stride: tl.int32, w_in_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)

        # Reduce over input H
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            y_ptrs = y_base + h_in_offsets * y_h_stride
            y_vals = tl.load(y_ptrs, mask=h_in_mask, other=0.0)

            w_ptrs = W_ptr + h_out * w_out_stride + h_in_offsets * w_in_stride
            w_vals = tl.load(w_ptrs, mask=h_in_mask, other=0.0)

            acc += tl.sum(y_vals * w_vals, axis=0)

        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + h_out * out_h_stride
        tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure inputs are float32 for numerical consistency with original
        device = x.device
        x = x.to(torch.float32)
        in_proj_weight = in_proj_weight.to(torch.float32)
        in_proj_bias = in_proj_bias.to(torch.float32)
        conv_weight = conv_weight.to(torch.float32)
        conv_bias = conv_bias.to(torch.float32)
        out_proj_weight = out_proj_weight.to(torch.float32)
        out_proj_bias = out_proj_bias.to(torch.float32)

        B, S, H = x.shape
        I = 3 * H

        # 1) Triton in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=device, dtype=torch.float32)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Reconstruct B, C, x_proj via slicing (PyTorch views)
        B_tensor = BCx[:, :, :H]               # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]          # (B, S, H)
        x_proj_tensor = BCx[:, :, 2 * H:]      # (B, S, H)

        # 3) Elementwise gating: Bx = B * x_proj (torch, lightweight)
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 4) Grouped causal conv with kernel_size=4, groups=H
        # Transpose Bx to (B, H, S) for conv
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, S, H) -> (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight, conv_bias, conv_out,
            B, S, H,
            K=4,
            BLOCK_T=256,
            num_warps=4,
        )

        # 5) Output gating: y = C * conv_out
        # C_tensor: (B, S, H), conv_out: (B, H, S)
        conv_out_T = conv_out.transpose(1, 2)  # (B,H,S) -> (B,S,H)
        y = C_tensor * conv_out_T              # (B,S,H), float32

        # 6) Triton out_proj: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B,S,H)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
