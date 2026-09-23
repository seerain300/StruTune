import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I,) or None
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,   # batch size
    S: tl.int32,   # sequence length
    H: tl.int32,   # hidden size
    I: tl.int32,   # output channels = 3*H
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    HAS_BIAS: tl.int32,       # 1 if Bias_ptr is valid, else 0
    BLOCK_H: tl.constexpr,    # tile size along H
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
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
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            # dot product over the tile
            acc += tl.sum(x_vals * w_vals, axis=0)

        if HAS_BIAS:
            bias_i = tl.load(Bias_ptr + i)
            acc += bias_i

        # store result; Out_ptr is expected to be float32
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *const float, input Bx: (B, H, S), pre-padded with zeros left
    W_ptr,          # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const float, conv_bias: (H)
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32,    # batch size
    S: tl.int32,    # sequence length
    H: tl.int32,    # hidden size (groups)
    K: tl.int32,    # kernel size (4)
    # strides
    bx_b_stride: tl.int32, bx_g_stride: tl.int32, bx_t_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # Grid: axis=0 over B*H (group) and axis=1 over tiles of S
    pid_group = tl.program_id(axis=0)  # 0..(B*H-1)
    pid_tile = tl.program_id(axis=1)
    b = pid_group // H
    g = pid_group % H

    # tile offsets along S
    t_start = pid_tile * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < S

    # base pointers
    bx_base = Bx_ptr + b * bx_b_stride + g * bx_g_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # accumulate over K=4
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # for causal conv with K=4: output at t uses Bx[t + k - 1], k=0..3
    for k in range(0, 4):
        t_in = t_offsets + k - 1
        in_mask = (t_in >= 0) & (t_in < S) & t_mask
        vals = tl.load(
            bx_base + t_in * bx_t_stride,
            mask=in_mask,
            other=0.0
        ).to(tl.float32)

        w_k = tl.load(W_ptr + g * w_g_stride + k * w_k_stride).to(tl.float32)
        acc += vals * w_k

    # add bias
    bias_g = tl.load(Bias_ptr + g).to(tl.float32)
    acc += bias_g

    # store
    tl.store(out_base + t_offsets * out_t_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    W_ptr,          # *const float, out_proj_weight: (H, H)
    Bias_ptr,       # *const float, out_proj_bias: (H)
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,    # batch size
    S: tl.int32,    # sequence length
    H: tl.int32,    # hidden size
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_out_h_stride: tl.int32, w_out_in_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # reduce over input H dimension
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_in_offsets < H

            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                W_ptr + h_out * w_out_h_stride + h_in_offsets * w_out_in_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        # add bias
        bias_h = tl.load(Bias_ptr + h_out).to(tl.float32)
        acc += bias_h

        # store
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,               # (B, S, H)
        in_proj_weight: torch.Tensor,  # (I, H), I=3*H
        in_proj_bias: torch.Tensor,    # (I,)
        conv_weight: torch.Tensor,     # (H, 1, 4)
        conv_bias: torch.Tensor,       # (H,)
        out_proj_weight: torch.Tensor, # (H, H)
        out_proj_bias: torch.Tensor,   # (H,)
    ):
        # in_proj: (B, S, H) -> (B, S, I) via F.linear
        B, S, H = x.shape
        I = 3 * H

        # Ensure dtype float32 for Triton kernels; keep inputs contiguous
        x_fp32 = x.to(torch.float32).contiguous()
        w_in_fp32 = in_proj_weight.to(torch.float32).contiguous()
        bias_in_fp32 = in_proj_bias.to(torch.float32).contiguous() if in_proj_bias is not None else None

        bcx = torch.empty((B, S, I), dtype=torch.float32, device=x.device)

        # launch in_proj kernel
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_fp32, w_in_fp32, bias_in_fp32 if bias_in_fp32 is not None else torch.empty(1, device=x.device, dtype=torch.float32),
            bcx,
            B, S, H, I,
            x_fp32.stride(0), x_fp32.stride(1), x_fp32.stride(2),
            w_in_fp32.stride(0), w_in_fp32.stride(1),
            bcx.stride(0), bcx.stride(1), bcx.stride(2),
            1 if (bias_in_fp32 is not None) else 0,
            BLOCK_H=64,
            num_warps=4,
        )

        # Now split BCx into B, C, x_proj along channel dimension I=3*H:
        # Each has shape: (B, H, S)
        # PyTorch view/slice for correctness
        B_tensor = bcx[:, :, :H]
        C_tensor = bcx[:, :, H:2 * H]
        x_proj_tensor = bcx[:, :, 2 * H:]

        # Element-wise gating: Bx = B * x_proj
        Bx = (B_tensor * x_proj_tensor).to(torch.float32).contiguous()  # (B, H, S)

        # Grouped causal conv with K=4, groups=H:
        # conv_weight: (H, 1, 4), conv_bias: (H), input Bx: (B, H, S)
        # Note: no torch.pad; handle causal via masked loads in Triton
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        w_conv_fp32 = conv_weight.to(torch.float32).contiguous()
        bias_conv_fp32 = conv_bias.to(torch.float32).contiguous()

        grid_conv = (B * H, triton.cdiv(S, 128))  # tile S along axis=1
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, w_conv_fp32, bias_conv_fp32, conv_out,
            B, S, H, 4,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            w_conv_fp32.stride(0), w_conv_fp32.stride(3),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=128,
            num_warps=4,
        )

        # Output gating with C: y = C * conv_out, shapes: (B, H, S)
        y = (C_tensor * conv_out).to(torch.float32).contiguous()

        # Final output projection: (B, S, H) = y @ out_proj_weight^T + out_proj_bias
        out = torch.empty((B, S, H), dtype=torch.float32, device=x.device)

        w_out_fp32 = out_proj_weight.to(torch.float32).contiguous()
        bias_out_fp32 = out_proj_bias.to(torch.float32).contiguous()

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, w_out_fp32, bias_out_fp32, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            w_out_fp32.stride(0), w_out_fp32.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return out


# Example usage (not part of evaluation):
# model = ModelNew().cuda()
# x = torch.randn(2, 4096, 256, device='cuda', dtype=torch.float32)
# in_proj_weight = torch.randn(3*256, 256, device='cuda', dtype=torch.float32)
# in_proj_bias = torch.randn(3*256, device='cuda', dtype=torch.float32)
# conv_weight = torch.randn(256, 1, 4, device='cuda', dtype=torch.float32)
# conv_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# out_proj_weight = torch.randn(256, 256, device='cuda', dtype=torch.float32)
# out_proj_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# y = model(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


def run(*args):
    return ModelNew()(*args)
