import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I) or nullptr
    Out_ptr,       # *float, output BCx: (B, S, I)
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
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float, input Bx: (B, H, S), contiguous
    W_ptr,         # *const float, conv_weight: (H, 1, 4), contiguous
    Bias_ptr,      # *const float, conv_bias: (H) or nullptr
    Out_ptr,       # *float, output conv_out: (B, H, S), contiguous
    B: tl.int32,
    H: tl.int32,   # group dimension
    S: tl.int32,
    K: tl.int32,   # kernel_size = 4
    has_bias: tl.int32,
    # Note: Bx is (B, H, S) contiguous; strides are effectively
    # bx_b_stride = H*S, bx_h_stride = S, bx_s_stride = 1
    # out is (B, H, S) contiguous similarly
    BLOCK_T: tl.constexpr,
):
    # Grid: one program per (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # Base pointers
    Bx_base = Bx_ptr + b * H * S + g * S
    Out_base = Out_ptr + b * H * S + g * S

    # Causal padding: left=K-1=3, right=0
    for t in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # sum over k=0..K-1
        for k in range(0, K):
            in_pos = t + k - (K - 1)  # shift by -3 for causal left pad
            if in_pos >= 0 and in_pos < S:
                x_val = tl.load(Bx_base + in_pos).to(tl.float32)
            else:
                x_val = tl.zeros((), dtype=tl.float32)
            # load conv weight for group g, k
            w_val = tl.load(W_ptr + g * 4 + k).to(tl.float32)
            acc += x_val * w_val
        if has_bias:
            bval = tl.load(Bias_ptr + g).to(tl.float32)
            acc += bval
        tl.store(Out_base + t, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Bias_ptr,      # *const float, out_proj_bias: (H) or nullptr
    Out_ptr,       # *float, output: (B, S, H)
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


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 for compute and outputs to match reference numerics
        x = x.to(torch.float32)
        in_proj_weight = in_proj_weight.to(torch.float32)
        in_proj_bias = in_proj_bias.to(torch.float32) if in_proj_bias is not None else None
        conv_weight = conv_weight.to(torch.float32)
        conv_bias = conv_bias.to(torch.float32) if conv_bias is not None else None
        out_proj_weight = out_proj_weight.to(torch.float32)
        out_proj_bias = out_proj_bias.to(torch.float32) if out_proj_bias is not None else None

        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj: linear (B, S, H) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)
        # Allocate bias buffer if needed (kernel expects Bias_ptr even if None, pass zeros)
        has_bias_in = 1 if (in_proj_bias is not None) else 0
        bias_in_ptr = in_proj_bias if (in_proj_bias is not None) else torch.zeros(I, device=x.device, dtype=torch.float32)
        in_proj_linear_kernel[(B * S,)](
            x, in_proj_weight, bias_in_ptr, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            has_bias_in,
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx into B, C, x_proj
        # BCx: (B, S, I), I=3*H -> (B, H, S), (B, H, S), (B, H, S)
        # We need contiguous (B, H, S) view for conv. PyTorch allows slicing for views.
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # Elementwise gating: Bx = B * x_proj (PyTorch lightweight op)
        Bx = B_tensor * x_proj_tensor  # (B, H, S)

        # 3) Grouped causal conv: Bx (B, H, S) -> conv_out (B, H, S), kernel_size=4, groups=H
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        has_bias_conv = 1 if (conv_bias is not None) else 0
        bias_conv_ptr = conv_bias if (conv_bias is not None) else torch.zeros(H, device=x.device, dtype=torch.float32)
        # Launch one program per (b, g)
        grid = (B * H,)
        grouped_causal_conv1d_kernel[grid](
            Bx, conv_weight, bias_conv_ptr, conv_out,
            B, H, S, 4,
            has_bias_conv,
            BLOCK_T=128,
            num_warps=4,
        )

        # 4) Output gating: y = C * conv_out
        # conv_out: (B, H, S), C_tensor: (B, H, S)
        y = C_tensor * conv_out  # (B, H, S)

        # Transpose y to (B, S, H) for out_proj
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)

        # 5) out_proj: linear (B, S, H) -> (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        has_bias_out = 1 if (out_proj_bias is not None) else 0
        bias_out_ptr = out_proj_bias if (out_proj_bias is not None) else torch.zeros(H, device=x.device, dtype=torch.float32)
        out_proj_linear_kernel[(B * S,)](
            y_T, out_proj_weight, bias_out_ptr, output,
            B, S, H,
            y_T.stride(0), y_T.stride(1), y_T.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            has_bias_out,
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
