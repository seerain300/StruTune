import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I) or None
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

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + i * W_ptr.strided(1) + h_offsets * W_ptr.strided(1), mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(x_vals * w_vals, axis=0)
        # add bias if provided
        if Bias_ptr is not None:
            bval = tl.load(Bias_ptr + i).to(tl.float32)
            acc += bval
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,         # *const float, input Bx: (B, H, S)
    W_ptr,          # *const float, conv_weight: (H, 1, K), K=4
    Bias_ptr,       # *const float, conv_bias: (H)
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H_groups: tl.int32,  # number of groups along H, here H
    S: tl.int32,
    K: tl.int32,         # kernel_size, 4
    # strides for Bx
    bx_b_stride: tl.int32, bx_g_stride: tl.int32, bx_t_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # 2D grid: axis0 over B*S tiles, axis1 over groups
    axis0 = tl.program_id(axis=0)
    g = tl.program_id(axis=1)

    # number of tiles along S
    num_tiles = (S + BLOCK_T - 1) // BLOCK_T
    tile_id = axis0 % num_tiles
    b = axis0 // num_tiles

    start = tile_id * BLOCK_T
    t_offsets = start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < S

    # base pointers
    bx_base = Bx_ptr + b * bx_b_stride + g * bx_g_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # accum for this tile
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # conv: out[t] = sum_k input[t + k - 1] * w[g, 0, k] + bias[g]
    for k in range(0, K):
        t_k = t_offsets + k - 1
        pos_mask = (t_k >= 0) & (t_k < S) & t_mask
        vals = tl.load(bx_base + t_k * bx_t_stride, mask=pos_mask, other=0.0).to(tl.float32)

        # weight (H, 1, K) -> scalar per group k
        w_val = tl.load(W_ptr + g * W_ptr.strided(0) + 0 * W_ptr.strided(1) + k * W_ptr.strided(2)).to(tl.float32)
        acc += vals * w_val

    # add bias
    bval = tl.load(Bias_ptr + g).to(tl.float32)
    acc += bval

    # store results
    tl.store(out_base + t_offsets * out_t_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float, input y: (B, S, H)
    W_ptr,          # *const float, out_proj_weight: (H, H)
    Bias_ptr,       # *const float, out_proj_bias: (H) or None
    Out_ptr,        # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
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
            h_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + h_out * W_ptr.strided(0) + h_offsets * W_ptr.strided(1), mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        # add bias if provided
        if Bias_ptr is not None:
            bval = tl.load(Bias_ptr + h_out).to(tl.float32)
            acc += bval
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        1) in_proj: x -> (B, S, 3*H)
        2) Split: (B, H), (C, H), (x_proj, H) from step 1
        3) Bx = B * x_proj (elementwise)
        4) Grouped causal conv: Bx (B, H, S) -> conv_out (B, H, S), groups=H, kernel_size=4
        5) y = C * conv_out (elementwise)
        6) out_proj: y -> final (B, S, H)
        """
        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        x_c = x.contiguous()
        w_in_c = in_proj_weight.contiguous()
        BCx = torch.empty((B, S, I), device=x.device, dtype=x.dtype)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_c, w_in_c, (in_proj_bias if in_proj_bias is not None else x_c), BCx,
            B, S, H, I,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split into B_tensor, C_tensor, x_proj_tensor (PyTorch slicing is lightweight and exact)
        B_tensor = BCx[:, :, :H].contiguous()
        C_tensor = BCx[:, :, H:2 * H].contiguous()
        x_proj_tensor = BCx[:, :, 2 * H:].contiguous()

        # Elementwise gate: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 3) Grouped causal conv: input Bx (B, H, S), weight conv_weight (H, 1, 4), bias conv_bias (H), groups=H
        # Transpose to (B, H, S)
        Bx_t = Bx.transpose(1, 2).contiguous()  # (B, S, H) -> (B, H, S)
        conv_weight_c = conv_weight.contiguous()  # (H, 1, 4)
        conv_bias_c = conv_bias.contiguous() if conv_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        # Kernel launch grid: (num_tiles * B, H)
        BLOCK_T = 256
        num_tiles = (S + BLOCK_T - 1) // BLOCK_T
        grid_conv = (num_tiles * B, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_t, conv_weight_c, conv_bias_c, conv_out,
            B, H, S, 4,
            Bx_t.stride(0), Bx_t.stride(1), Bx_t.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=BLOCK_T,
            num_warps=4,
        )

        # 4) Output gate: y = C_tensor * conv_out
        conv_out_t = conv_out.transpose(1, 2).contiguous()  # (B, H, S) -> (B, S, H)
        y = C_tensor * conv_out_t

        # 5) out_proj: y @ out_proj_weight^T + out_proj_bias -> final output (B, S, H)
        y_c = y.contiguous()
        w_out_c = out_proj_weight.contiguous()
        out_bias_c = out_proj_bias.contiguous() if out_proj_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y_c, w_out_c, (out_bias_c if out_bias_c.numel() > 0 else y_c), out,
            B, S, H,
            y_c.stride(0), y_c.stride(1), y_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
