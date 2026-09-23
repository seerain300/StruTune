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
    I: tl.int32,   # output channels = 3 * H
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # grid: axis=0 over B, axis=1 over S
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    # base pointers for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels I
    for i in range(0, I):
        acc = 0.0  # accumulate in float32
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            # weight row for output channel i: W[i, :]
            w_vals = tl.load(W_ptr + i * W_ptr.shape[0] + h_offsets * W_ptr.shape[1], mask=h_mask, other=0.0)
            acc += tl.sum(x_vals * w_vals, axis=0)
        # store result to Out[b, s, i]
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,          # *const float, input Bx: (B, H, S)
    W_ptr,           # *const float, conv_weight: (H, 1, 4) per group
    BIAS_ptr,        # *const float, conv_bias: (H)
    Out_ptr,         # *float, output conv_out: (B, H, S)
    B: tl.int32,     # batch size (unused but kept)
    H: tl.int32,     # channels per group
    S: tl.int32,     # sequence length
    # strides for Bx
    bx_b_stride: tl.int32, bx_g_stride: tl.int32, bx_t_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,  # tile size along sequence (compile-time)
):
    # grid: axis=0 over B, axis=1 over H
    b = tl.program_id(axis=0)
    g = tl.program_id(axis=1)

    # base pointers for this (b, g)
    bx_base = Bx_ptr + b * bx_b_stride + g * bx_g_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # iterate over output sequence positions in tiles
    for t_start in range(0, S, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        # causal conv with kernel_size=4 and padding=3
        # conv_out[g, t] = sum_{k=0..3} Bx[g, t + k - 1] * W[g, k] + bias[g]
        for k in range(4):
            pos = t_offsets + k - 1
            pos_mask = (pos >= 0) & (pos < S) & t_mask
            bx_vals = tl.load(bx_base + pos * bx_t_stride, mask=pos_mask, other=0.0)
            # weight for group g, kernel position k
            w_val = tl.load(W_ptr + g * W_ptr.shape[0] + 0 * W_ptr.shape[1] + k * W_ptr.shape[2])
            acc += bx_vals * w_val
        # add bias
        bias_val = tl.load(BIAS_ptr + g)
        acc += bias_val
        # store result
        tl.store(out_base + t_offsets * out_t_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    Wout_ptr,       # *const float, out_proj_weight: (H, H)
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for Y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # grid: axis=0 over B, axis=1 over S
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = 0.0
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=h_in_mask, other=0.0)
            w_vals = tl.load(Wout_ptr + h_out * Wout_ptr.shape[0] + h_in_offsets * Wout_ptr.shape[1], mask=h_in_mask, other=0.0)
            acc += tl.sum(y_vals * w_vals, axis=0)
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure contiguous and cast to float32 for stable and consistent compute
        x = x.contiguous()
        x = x.to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else None
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else None

        B, S, H = x.shape
        I = 3 * H  # in_proj output channels

        # 1) in_proj linear: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), dtype=torch.float32, device=x.device)

        # launch Triton kernel
        BLOCK_H = 64
        grid_in = (B, S)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        # Shapes: (B, H, S), (B, H, S), (B, H, S)
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, H, S)
        Bx = Bx.contiguous()  # ensure contiguous for conv

        # 4) Grouped causal 1D conv on Bx: (B, H, S) with groups=H, kernel_size=4, padding=3
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        grid_conv = (B, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_weight, conv_bias if conv_bias is not None else conv_weight.new_zeros(H), conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256,  # tile along S
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out
        # Note: C_tensor shape is (B, H, S), conv_out shape is (B, H, S)
        C_t = C_tensor.transpose(1, 2)  # (B, S, H)
        y = C_t * conv_out  # elementwise multiply (B, S, H)

        # 6) Final output projection: y -> output via out_proj
        output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_out = (B, S)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
