import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,        # *const float, input x: (B, S, H)
    W_ptr,        # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,      # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for X
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # one program per (b, s)
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
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            x_vals = x_vals.to(tl.float32)
            w_vals = tl.load(W_ptr + i * W_ptr.strides(0) + h_offsets * W_ptr.strides(1), mask=h_mask, other=0.0)
            w_vals = w_vals.to(tl.float32)
            acc += tl.sum(x_vals * w_vals, axis=0)
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,        # *const float, input y: (B, S, H)
    Wout_ptr,     # *const float, out_proj_weight: (H, H)
    Bias_ptr,     # *const float, out_proj_bias: (H)
    Out_ptr,      # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for Y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Wout
    wout_h0_stride: tl.int32, wout_h1_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H, BLOCK_H):
        h_out_offsets = h_out + tl.arange(0, BLOCK_H)
        h_mask = h_out_offsets < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            in_mask = h_in_offsets < H
            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=in_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(Wout_ptr + h_out_offsets[:, None] * wout_h0_stride + h_in_offsets[None, :] * wout_h1_stride, mask=in_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(y_vals[:, None] * w_vals, axis=1)

        bias_vals = tl.load(Bias_ptr + h_out_offsets, mask=h_mask, other=0.0).to(tl.float32)
        acc += bias_vals
        tl.store(out_base + h_out_offsets * out_h_stride, acc, mask=h_mask)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,       # *const float, input after gating: (B, H, S)
    Weight_ptr,   # *const float, conv_weight: (H, 1, 4)
    Bias_ptr,     # *const float, conv_bias: (H)
    Out_ptr,      # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    K: tl.int32,
    # strides for Bx
    bx_b_stride: tl.int32, bx_h_stride: tl.int32, bx_s_stride: tl.int32,
    # strides for Weight
    w_h_stride: tl.int32, w_c_stride: tl.int32, w_k_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_S: tl.constexpr,
):
    # grid: axis0 over B*H, axis1 over tiles of S
    pid0 = tl.program_id(axis=0)
    b = pid0 // H
    g = pid0 % H

    pid1 = tl.program_id(axis=1)
    s_start = pid1 * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # causal padding: pad left by K-1
    # For each k in [0..K-1], compute indices t_in = s_offsets + k - (K-1)
    for k in range(0, K):
        t_in = s_offsets + k - (K - 1)
        # mask for valid input positions (within [0, S))
        mask_in = (t_in >= 0) & (t_in < S) & s_mask

        # load Bx[b, g, t_in]
        bx_ptr = Bx_ptr + b * bx_b_stride + g * bx_h_stride + t_in * bx_s_stride
        bx_vals = tl.load(bx_ptr, mask=mask_in, other=0.0).to(tl.float32)

        # load weight[g, 0, k]
        w_ptr = Weight_ptr + g * w_h_stride + 0 * w_c_stride + k * w_k_stride
        w_val = tl.load(w_ptr).to(tl.float32)

        acc += bx_vals * w_val

    # add bias[g]
    bias_val = tl.load(Bias_ptr + g).to(tl.float32)
    acc += bias_val

    # store conv_out[b, g, s_offsets]
    out_ptr = Out_ptr + b * out_b_stride + g * out_h_stride + s_offsets * out_s_stride
    tl.store(out_ptr, acc, mask=s_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        """
        x: (B, S, H)
        in_proj_weight: (I, H), I=3*H
        in_proj_bias: (I,)
        conv_weight: (H, 1, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        Returns: output (B, S, H)
        """
        B, S, H = x.shape
        I = in_proj_weight.shape[0]
        assert in_proj_weight.shape[1] == H, "in_proj_weight second dim must be H"
        assert conv_weight.shape[0] == H and conv_weight.shape[1] == 1 and conv_weight.shape[2] == 4, "conv_weight must be (H, 1, 4)"
        assert out_proj_weight.shape[0] == H and out_proj_weight.shape[1] == H, "out_proj_weight must be (H, H)"
        assert conv_bias is not None and conv_bias.shape[0] == H, "conv_bias must be shape (H,)"
        assert out_proj_bias is not None and out_proj_bias.shape[0] == H, "out_proj_bias must be shape (H,)"

        # Ensure contiguous tensors
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        conv_weight = conv_weight.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=x.dtype)

        BLOCK_H = 64
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # 2) Slice BCx: B_tensor, C_tensor, x_proj_tensor
        # BCx: (B, S, I) with I=3*H
        B_tensor = BCx[:, :, :H].contiguous()         # (B, S, H)
        C_tensor = BCx[:, :, H:2*H].contiguous()      # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:].contiguous()  # (B, S, H)

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor                 # (B, S, H)

        # 4) Transpose for conv: (B, H, S)
        Bx_t = Bx.transpose(1, 2).contiguous()        # (B, S, H) -> (B, H, S)

        # 5) Grouped causal conv: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        BLOCK_S = 128
        grid_conv = (B * H, triton.cdiv(S, BLOCK_S))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_t, conv_weight, conv_bias, conv_out,
            B, H, S, 4,
            Bx_t.stride(0), Bx_t.stride(1), Bx_t.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        # 6) Output gating: y = C_tensor.transpose(-1, -2) * conv_out
        C_trans = C_tensor.transpose(1, 2).contiguous()  # (B, H, S)
        y = C_trans * conv_out                           # (B, H, S)

        # 7) out_proj: output (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y.transpose(1, 2).contiguous(),      # y (B, H, S) -> (B, S, H)
            out_proj_weight, out_proj_bias,
            output,
            B, S, H,
            y.transpose(1, 2).stride(0), y.transpose(1, 2).stride(1), y.transpose(1, 2).stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )
        return output


def run(*args):
    return ModelNew()(*args)
