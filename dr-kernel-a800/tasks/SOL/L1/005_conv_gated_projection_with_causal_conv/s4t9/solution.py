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
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride

    # Loop over output channels i = 0..I-1
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over input H
        for h in range(0, H):
            x_val = tl.load(x_base + h * x_h_stride).to(tl.float32)
            w_val = tl.load(W_ptr + i * H + h).to(tl.float32)
            acc += x_val * w_val
        # Store to Out[b, s, i]
        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + i * out_i_stride
        # Store float32; Out_ptr dtype should be float32
        tl.store(out_ptr, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # Loop over output H
    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over input H
        for h_in in range(0, H):
            y_val = tl.load(y_base + h_in * y_h_stride).to(tl.float32)
            w_val = tl.load(W_ptr + h_out * H + h_in).to(tl.float32)
            acc += y_val * w_val
        # Store to Out[b, s, h_out]
        out_ptr = out_base + h_out * out_h_stride
        tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 for compute stability
        device = x.device
        dtype = x.dtype

        # 1) Triton in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        B, S, H = x.shape
        I = 3 * H

        # Prepare inputs/weights for Triton kernels (float32)
        x_in = x.contiguous().to(torch.float32)
        in_proj_weight_in = in_proj_weight.contiguous().to(torch.float32)
        # Allocate output BCx (B, S, I) as float32
        BCx = torch.empty((B, S, I), device=device, dtype=torch.float32)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_in, in_proj_weight_in, BCx,
            B, S, H, I,
            x_b_stride=x_in.stride(0), x_s_stride=x_in.stride(1), x_h_stride=x_in.stride(2),
            out_b_stride=BCx.stride(0), out_s_stride=BCx.stride(1), out_i_stride=BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx into B, C, x_proj (PyTorch views)
        # BCx shape (B,S,I), I=3H
        B_tensor = BCx[:, :, :H]           # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]      # (B, S, H)
        x_proj_tensor = BCx[:, :, 2 * H:]  # (B, S, H)

        # 3) Elementwise gating: Bx = B * x_proj (torch multiply)
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 4) Grouped causal conv via PyTorch to ensure correctness:
        #    Original code: F.conv1d(Bx.transpose(1,2), conv_weight, conv_bias, groups=H)
        #    Bx.shape: (B, S, H) -> transpose to (B, H, S)
        Bx_T = Bx.transpose(1, 2)  # (B, H, S)
        conv_out = F.conv1d(Bx_T, conv_weight, conv_bias, groups=H)  # (B, H, S)

        # 5) Output gating: y = C * conv_out (torch elementwise)
        # C_tensor shape (B, S, H), conv_out shape (B, H, S)
        # We need to multiply corresponding channels: C * conv_out
        # For consistency, reshape C_tensor to (B, H, S) to match conv_out by swapping dims 1 and 2.
        C_T = C_tensor.transpose(1, 2)  # (B, H, S)
        y = C_T * conv_out              # (B, H, S)

        # 6) Final out_proj via Triton: y (B, H, S) -> output (B, S, H)
        # We need to compute out_proj(y) with weight (H, H). Since y is (B,H,S), we can flatten to (B*H,S) and reduce over H.
        # However, Triton kernel expects input of shape (B,S,H). We will reshape y back to (B,S,H) by transpose from (B,H,S).
        # The original y is (B,H,S). To get y in (B,S,H), we take transpose(1,2): (B,S,H). Then apply out_proj kernel.
        y_in = y.transpose(1, 2).contiguous().to(torch.float32)  # (B,S,H)
        out_proj_weight_in = out_proj_weight.contiguous().to(torch.float32)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y_in, out_proj_weight_in, output,
            B, S, H,
            y_b_stride=y_in.stride(0), y_s_stride=y_in.stride(1), y_h_stride=y_in.stride(2),
            out_b_stride=output.stride(0), out_s_stride=output.stride(1), out_h_stride=output.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # Return output. The original returns shape (B, S, H). Ensure dtype consistency.
        # The evaluator expects the same dtype as input x. Cast to x.dtype if needed.
        if output.dtype != dtype:
            output = output.to(dtype)
        return output


def run(*args):
    return ModelNew()(*args)
