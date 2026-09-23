import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const T, input x: (B, S, H)
    W_ptr,         # *const T, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const T, in_proj_bias: (I) or nullptr
    Out_ptr,       # *T, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
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
        if has_bias:
            bval = tl.load(Bias_ptr + i).to(tl.float32)
            acc += bval
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const T, input y: (B, S, H)
    W_ptr,         # *const T, out_proj_weight: (H, H)
    Bias_ptr,      # *const T, out_proj_bias: (H) or nullptr
    Out_ptr,       # *T, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_h_out_stride: tl.int32, w_h_in_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
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
                W_ptr + h_out * w_h_out_stride + h_in_offsets * w_h_in_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        if has_bias:
            bval = tl.load(Bias_ptr + h_out).to(tl.float32)
            acc += bval
        tl.store(out_base + h_out * out_h_stride, acc)


@triton.jit
def gated_mul_kernel(
    A_ptr, B_ptr, Out_ptr,  # A, B are (B*S*H) contiguous
    N: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = a * b
    tl.store(Out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Implements the original flow:
          1) BCx = in_proj(x) via F.linear, but compute in Triton for (B, S, I), I=3*H.
          2) Slice BCx into B_tensor, C_tensor, x_proj_tensor.
          3) Bx = B_tensor * x_proj_tensor (elementwise gating) computed in Triton.
          4) conv_out = F.conv1d(Bx.transpose(-1, -2), conv_weight, conv_bias, groups=H, padding=K-1)
          5) y = C_tensor * conv_out.transpose(-1, -2) (elementwise gating) computed in Triton.
          6) output = out_proj(y) via F.linear, but compute in Triton for (B, S, H).
        Triton kernels are invoked for in_proj, out_proj, and elementwise gating to satisfy the requirement.
        """
        # Shapes
        Bsz, S, H = x.shape
        I = 3 * H
        K = conv_weight.shape[2]

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        x_in = x.contiguous()
        W_in = in_proj_weight.contiguous()
        if in_proj_bias is None:
            Bias_in = torch.zeros(I, device=x.device, dtype=x.dtype)
        else:
            Bias_in = in_proj_bias.contiguous()
        BCx = torch.empty((Bsz, S, I), device=x.device, dtype=x.dtype)

        BLOCK_H_in = 64
        grid_in = (Bsz * S,)
        in_proj_linear_kernel[grid_in](
            x_in, W_in, Bias_in,
            BCx,
            Bsz, S, H, I,
            x_in.stride(0), x_in.stride(1), x_in.stride(2),
            W_in.stride(0), W_in.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            int(Bias_in is not None),
            BLOCK_H=BLOCK_H_in,
            num_warps=4
        )

        # 2) Slicing
        B_tensor = BCx[:, :, :H]          # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]       # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:]   # (B, S, H)

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor (Triton)
        Bx = torch.empty_like(B_tensor)   # temporary to hold result
        N1 = Bsz * S * H
        BLOCK1 = 1024
        grid1 = ((N1 + BLOCK1 - 1) // BLOCK1,)
        # Flatten for kernel
        Bx_flat = Bx.reshape(-1)
        B_flat = B_tensor.reshape(-1).contiguous()
        x_proj_flat = x_proj_tensor.reshape(-1).contiguous()
        gated_mul_kernel[grid1](
            B_flat, x_proj_flat, Bx_flat,
            N1,
            BLOCK=BLOCK1,
            num_warps=4
        )

        # 4) Grouped causal conv: conv_out = F.conv1d(Bx.transpose(-1, -2), conv_weight, conv_bias, groups=H, padding=K-1)
        # Input for conv: (B, H, S)
        Bx_conv = Bx.transpose(-1, -2).contiguous()   # (B, H, S)
        conv_out = torch.nn.functional.conv1d(Bx_conv, conv_weight, conv_bias, groups=H, padding=K - 1)

        # 5) Output gating: y = C_tensor * conv_out.transpose(-1, -2) (Triton)
        y = torch.empty_like(C_tensor)               # (B, S, H)
        N2 = Bsz * S * H
        BLOCK2 = 1024
        grid2 = ((N2 + BLOCK2 - 1) // BLOCK2,)
        y_flat = y.reshape(-1)
        C_flat = C_tensor.reshape(-1).contiguous()
        conv_out_t_flat = conv_out.transpose(-1, -2).reshape(-1).contiguous()
        gated_mul_kernel[grid2](
            C_flat, conv_out_t_flat, y_flat,
            N2,
            BLOCK=BLOCK2,
            num_warps=4
        )

        # 6) out_proj: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        W_out = out_proj_weight.contiguous()
        if out_proj_bias is None:
            Bias_out = torch.zeros(H, device=y.device, dtype=y.dtype)
        else:
            Bias_out = out_proj_bias.contiguous()
        output = torch.empty((Bsz, S, H), device=y.device, dtype=y.dtype)

        BLOCK_H_out = 64
        grid_out = (Bsz * S,)
        out_proj_linear_kernel[grid_out](
            y, W_out, Bias_out,
            output,
            Bsz, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            W_out.stride(0), W_out.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            int(Bias_out is not None),
            BLOCK_H=BLOCK_H_out,
            num_warps=4
        )

        return output


def run(*args):
    return ModelNew()(*args)
