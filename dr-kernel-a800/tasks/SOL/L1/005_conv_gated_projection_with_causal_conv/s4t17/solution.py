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
    # strides for x: (B, S, H)
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out: (B, S, I)
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # reduce over H for each output channel i
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + i * W_ptr.itemsize + h_offsets * w_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            # Note: W_ptr.itemsize is not standard; instead, pass strides for W if needed. Here we assume contiguous W and stride along H is 1, but W is (I, H) so we can compute address as i*H + h_offsets
            # Better: treat W as contiguous (I*H elements) and compute address as i*H + h_offsets
            w_vals = tl.load(W_ptr + i * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(x_vals * w_vals, axis=0)
        # store acc to Out[b, s, i]
        # Out has strides (out_b_stride, out_s_stride, out_i_stride)
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    Wout_ptr,      # *const float, out_proj_weight: (H, H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y: (B, S, H)
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for out: (B, S, H)
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # for each output channel h_out, reduce over H
    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            # load Wout[h_out, h_offsets]
            w_vals = tl.load(Wout_ptr + h_out * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        tl.store(out_base + h_out * out_h_stride, acc)


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
        Triton-optimized forward:
        - in_proj via Triton kernel
        - conv via PyTorch (to avoid Triton runtime errors in grouped causal conv)
        - out_proj via Triton kernel
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj linear: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # x: (B, S, H), in_proj_weight: (I, H)
        # output BCx: (B, S, I), float32 compute, cast to x.dtype on store
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)
        # Prepare strides (assume contiguous)
        x_b_stride, x_s_stride, x_h_stride = x.stride()
        out_b_stride, out_s_stride, out_i_stride = BCx.stride()
        # Launch Triton
        # Choose BLOCK_H=64 for good performance; H=hidden_size is small in typical configs
        in_proj_linear_kernel[(B * S,)](
            x, in_proj_weight.to(torch.float32), BCx,
            B, S, H, I,
            x_b_stride, x_s_stride, x_h_stride,
            out_b_stride, out_s_stride, out_i_stride,
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx into B_tensor, C_tensor, x_proj_tensor
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        # Shapes: (B, H, S)
        Bx = B_tensor * x_proj_tensor
        Bx = Bx.transpose(-1, -2).contiguous()  # (B, H, S)

        # 4) Grouped causal conv in PyTorch
        # conv_weight: (H, 1, 4), conv_bias: (H), groups=H
        K = conv_weight.shape[2]
        padding = K - 1
        conv_out = torch.nn.functional.conv1d(
            Bx, conv_weight, conv_bias, groups=H, padding=padding
        )  # (B, H, S)

        # 5) Output gating: y = C * conv_out (C = C_tensor.transpose(-1, -2))
        C_t = C_tensor.transpose(-1, -2).contiguous()  # (B, H, S)
        y = C_t * conv_out  # (B, H, S)

        # 6) out_proj linear via Triton: output = F.linear(y, out_proj_weight, out_proj_bias)
        # y: (B, H, S) → (B, S, H) in kernel, we use y.T for kernel
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        # Strides for y.T
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)
        y_b_stride, y_s_stride, y_h_stride = y_T.stride()
        out_b_stride_out, out_s_stride_out, out_h_stride_out = output.stride()

        out_proj_linear_kernel[(B * S,)](
            y_T, out_proj_weight.to(torch.float32), output,
            B, S, H,
            y_b_stride, y_s_stride, y_h_stride,
            out_b_stride_out, out_s_stride_out, out_h_stride_out,
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
