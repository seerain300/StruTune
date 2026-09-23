import torch
import triton
import triton.language as tl

# Constants for tiling and types
BLOCK_H = 64  # tile size along hidden dimension
I_CONST = 3 * 256  # hardcoded I=3*H assuming H=256; we will pass actual H and I to Triton and use masks
OUT_I_TILES = 3  # number of tiles of I, since I_CONST = 3 * 256

@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # one program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels in tiles of I
    for tile_i in range(OUT_I_TILES):
        i_start = tile_i * BLOCK_H
        i_offsets = i_start + tl.arange(0, BLOCK_H)
        i_mask = i_offsets < I

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        # reduce over H in tiles
        for h_start in range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)  # (BLOCK_H,)
            # W layout is (I, H); load W rows for i_offsets over h_offsets
            w_vals = tl.load(
                W_ptr + i_offsets * w_i_stride + h_offsets * w_h_stride,
                mask=i_mask & h_mask,
                other=0.0
            ).to(tl.float32)  # (BLOCK_H,)

            acc += tl.sum(x_vals[:, None] * w_vals[None, :], axis=0)

        # store acc for this tile of I
        tl.store(out_base + i_offsets * out_i_stride, acc, mask=i_mask)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float, input for conv: (B, H, S) where S is sequence length
    W_ptr,         # *const float, conv_weight: (H, 1, 4), groups=H
    Bias_ptr,      # *const float, conv_bias: (H)
    Out_ptr,       # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    x_g_stride: tl.int32, x_t_stride: tl.int32,  # strides for Bx: (B, H, S)
    w_g_stride: tl.int32, w_k_stride: tl.int32,  # strides for W: (H, 1, 4)
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # one program per (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    x_base = Bx_ptr + b * x_g_stride + g * x_t_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # iterate over output positions t in tiles
    for t_start in range(0, S, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # K=4, causal padding: t+k-1 must be in [0, S)
        for k in range(4):
            t_k = t_offsets + k - 1
            in_bounds = (t_k >= 0) & (t_k < S) & t_mask

            x_vals = tl.load(
                x_base + t_k * x_t_stride,
                mask=in_bounds,
                other=0.0
            ).to(tl.float32)

            w_val = tl.load(
                W_ptr + g * w_g_stride + k * w_k_stride  # k index at innermost dim
            ).to(tl.float32)

            acc += x_vals * w_val

        # add bias
        bias_val = tl.load(Bias_ptr + g).to(tl.float32)
        acc += bias_val

        tl.store(out_base + t_offsets * out_t_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_out_ptr,     # *const float, out_proj_weight: (H, H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_out_o_stride: tl.int32, w_out_i_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # one program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H, BLOCK_H):
        h_out_offsets = h_out + tl.arange(0, BLOCK_H)
        h_out_mask = h_out_offsets < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                W_out_ptr + h_out_offsets * w_out_o_stride + h_in_offsets * w_out_i_stride,
                mask=h_out_mask & h_in_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(y_vals[:, None] * w_vals[None, :], axis=0)

        tl.store(out_base + h_out_offsets * out_h_stride, acc, mask=h_out_mask)


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
        # Ensure dtype is float32 and tensors are contiguous
        # This aligns with the original code which uses float32 by default
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj linear: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        # Allocate output BCx (float32, contiguous)
        BCx = torch.empty((B, S, I), dtype=torch.float32, device=x.device)

        # Launch in_proj_kernel
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        B_tensor = BCx[:, :, :H]   # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]  # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:]  # (B, S, H)

        # Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # float32

        # Transpose to (B, H, S) for conv1d with groups=H
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, S, H)
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        # 3) Grouped causal conv1d on Bx_trans using Triton
        # W shape: (H, 1, 4), bias: (H)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight, conv_bias, conv_out,
            B, H, S,
            Bx_trans.stride(0), Bx_trans.stride(1),  # x_g_stride=S, x_t_stride=1 for contiguous (B,S,H)
            conv_weight.stride(0), conv_weight.stride(2),  # w_g_stride=H, w_k_stride=4
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=BLOCK_H,  # tile along S; H=256 -> BLOCK_T=64
            num_warps=4,
            num_stages=2,
        )

        # 4) Output gating with C_tensor: y = C * conv_out
        # C_tensor: (B, S, H), conv_out: (B, H, S)
        # y: (B, S, H) after gating
        y = C_tensor.transpose(1, 2).contiguous() * conv_out  # (B, S, H)

        # 5) Final out_proj: y -> out_proj(y)
        output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
