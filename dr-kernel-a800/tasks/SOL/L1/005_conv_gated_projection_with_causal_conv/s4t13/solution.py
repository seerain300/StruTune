import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I,)
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32, S: tl.int32, H: tl.int32, I: tl.int32,
    # strides for x: (B, S, H)
    x_bs_stride: tl.int32, x_ss_stride: tl.int32, x_hs_stride: tl.int32,
    # strides for out: (B, S, I)
    out_bs_stride: tl.int32, out_ss_stride: tl.int32, out_is_stride: tl.int32,
    # strides for w: (I, H)
    w_is_stride: tl.int32, w_hs_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer for this (b, s) along H
    x_base = X_ptr + b * x_bs_stride + s * x_ss_stride

    # Loop over output channels i
    for i in range(0, I):
        acc = 0.0
        for h in range(0, H, BLOCK_H):
            h_idx = h + tl.arange(0, BLOCK_H)
            mask_h = h_idx < H
            # Load x[b, s, h:h+BLOCK_H]
            x_vals = tl.load(x_base + h_idx * x_hs_stride, mask=mask_h, other=0.0)
            # Load w[i, h:h+BLOCK_H]
            w_vals = tl.load(W_ptr + i * w_is_stride + h_idx * w_hs_stride, mask=mask_h, other=0.0)
            # Accumulate dot product
            acc += tl.sum(x_vals * w_vals, axis=0)
        # Add bias
        bias_val = tl.load(Bias_ptr + i)
        acc += bias_val
        # Store to Out[b, s, i]
        tl.store(Out_ptr + b * out_bs_stride + s * out_ss_stride + i * out_is_stride, acc)


@triton.jit
def pad1d_causal_kernel(
    Bx_ptr,        # *const float, input Bx: (B, H, S)
    BxP_ptr,       # *float, output Bx_padded: (B, H, S + K - 1), K=4
    B: tl.int32, H: tl.int32, S: tl.int32, K: tl.constexpr,
    # strides for Bx: (B, H, S)
    bx_bs_stride: tl.int32, bx_hs_stride: tl.int32, bx_ts_stride: tl.int32,
    # strides for Bx_padded: (B, H, S + K - 1)
    bxp_bs_stride: tl.int32, bxp_hs_stride: tl.int32, bxp_ts_stride: tl.int32,
):
    # 2D grid: axis=0 over B*H, axis=1 over S+K-1
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    b = pid0 // H
    h = pid0 % H
    t_out = pid1

    # map to source index
    t_src = t_out - (K - 1)
    valid = (t_src >= 0) & (t_src < S)
    # compute source and destination offsets
    bx_offset = b * bx_bs_stride + h * bx_hs_stride + t_src * bx_ts_stride
    bxp_offset = b * bxp_bs_stride + h * bxp_hs_stride + t_out * bxp_ts_stride

    val = tl.load(Bx_ptr + bx_offset, mask=valid, other=0.0)
    tl.store(BxP_ptr + bxp_offset, val)


@triton.jit
def grouped_causal_conv1d_kernel(
    BxP_ptr,       # *const float, Bx_padded: (B, H, S+K-1), K=4
    ConvW_ptr,     # *const float, conv_weight: (H, 1, 4) flattened as (H*4,)
    Bias_ptr,      # *const float, conv_bias: (H,)
    ConvOut_ptr,   # *float, conv_out: (B, H, S)
    B: tl.int32, H: tl.int32, S: tl.int32, K: tl.constexpr,
    # strides for BxP: (B, H, S+K-1)
    bxp_bs_stride: tl.int32, bxp_hs_stride: tl.int32, bxp_ts_stride: tl.int32,
    # strides for ConvOut: (B, H, S)
    co_bs_stride: tl.int32, co_gs_stride: tl.int32, co_ts_stride: tl.int32,
    conv_w_gs_stride: tl.int32, conv_w_ks_stride: tl.int32,
):
    # Grid: axis=0 over B*H, axis=1 over tiles of S
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    b = pid0 // H
    g = pid0 % H

    # Vectorize over output positions t in a tile of size BLOCK_T
    BLOCK_T = 256
    t_start = pid1 * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Accumulate over kernel window K=4 with causal padding
    # conv_weight is flattened to (H*4), so per g: weights = [ConvW[g*4 + 0], ConvW[g*4 + 1], ...]
    base_w = g * 4
    for k in range(K):
        w_k = tl.load(ConvW_ptr + base_w + k)
        t_in = t_offsets + (k - (K - 1))  # k - 3 for K=4
        valid = (t_in >= 0) & (t_in < (S + K - 1)) & mask_t
        # Load Bx_padded[b, g, t_in]
        bx_offset = b * bxp_bs_stride + g * bxp_hs_stride + t_in * bxp_ts_stride
        val = tl.load(BxP_ptr + bx_offset, mask=valid, other=0.0)
        acc += val * w_k

    # Add bias for group g
    bias_g = tl.load(Bias_ptr + g)
    acc += bias_g

    # Store conv_out[b, g, t_offsets]
    co_offset = b * co_bs_stride + g * co_gs_stride + t_offsets * co_ts_stride
    tl.store(ConvOut_ptr + co_offset, acc, mask=mask_t)


@triton.jit
def gated_mul_kernel_left(
    A_ptr, B_ptr, Out_ptr,
    Bsz: tl.int32, S: tl.int32, H: tl.int32,
    a_bs_stride: tl.int32, a_ss_stride: tl.int32, a_hs_stride: tl.int32,
    b_bs_stride: tl.int32, b_ss_stride: tl.int32, b_hs_stride: tl.int32,
    out_bs_stride: tl.int32, out_ss_stride: tl.int32, out_hs_stride: tl.int32,
):
    # 1D grid over B*S*H
    total = Bsz * S * H
    pid = tl.program_id(axis=0)
    # compute (b, s, h) from pid
    s = pid // (H)
    b = s // H
    h = pid % H
    # pointers
    a_ptr = A_ptr + b * a_bs_stride + s * a_ss_stride + h * a_hs_stride
    b_ptr = B_ptr + b * b_bs_stride + s * b_ss_stride + h * b_hs_stride
    out_ptr = Out_ptr + b * out_bs_stride + s * out_ss_stride + h * out_hs_stride
    a_val = tl.load(a_ptr)
    b_val = tl.load(b_ptr)
    tl.store(out_ptr, a_val * b_val)


@triton.jit
def gated_mul_kernel_right(
    A_ptr, B_ptr, Out_ptr,
    Bsz: tl.int32, S: tl.int32, H: tl.int32,
    a_bs_stride: tl.int32, a_ss_stride: tl.int32, a_hs_stride: tl.int32,
    b_bs_stride: tl.int32, b_ss_stride: tl.int32, b_hs_stride: tl.int32,
    out_bs_stride: tl.int32, out_ss_stride: tl.int32, out_hs_stride: tl.int32,
):
    # 1D grid over B*S*H
    total = Bsz * S * H
    pid = tl.program_id(axis=0)
    s = pid // H
    b = s // H
    h = pid % H
    a_ptr = A_ptr + b * a_bs_stride + s * a_ss_stride + h * a_hs_stride
    b_ptr = B_ptr + b * b_bs_stride + s * b_ss_stride + h * b_hs_stride
    out_ptr = Out_ptr + b * out_bs_stride + s * out_ss_stride + h * out_hs_stride
    a_val = tl.load(a_ptr)
    b_val = tl.load(b_ptr)
    tl.store(out_ptr, a_val * b_val)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    Wout_ptr,      # *const float, out_proj_weight: (H, H)
    Bias_ptr,      # *const float, out_proj_bias: (H,)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    # strides for y: (B, S, H)
    y_bs_stride: tl.int32, y_ss_stride: tl.int32, y_hs_stride: tl.int32,
    # strides for out: (B, S, H)
    out_bs_stride: tl.int32, out_ss_stride: tl.int32, out_hs_stride: tl.int32,
    # strides for Wout: (H, H)
    wout_oh_stride: tl.int32, wout_ih_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid over B*S
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    y_base = Y_ptr + b * y_bs_stride + s * y_ss_stride

    # For each output channel h_out
    for h_out in range(0, H, BLOCK_H):
        h_out_idx = h_out + tl.arange(0, BLOCK_H)
        mask_h_out = h_out_idx < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        # reduce over input H
        for h_in in range(0, H, BLOCK_H):
            h_in_idx = h_in + tl.arange(0, BLOCK_H)
            mask_h_in = h_in_idx < H
            # load y[b, s, h_in]
            y_vals = tl.load(y_base + h_in_idx * y_hs_stride, mask=mask_h_in, other=0.0)
            # load Wout[h_out_idx, h_in_idx]
            w_vals = tl.load(Wout_ptr + h_out_idx[:, None] * wout_oh_stride + h_in_idx[None, :] * wout_ih_stride, mask=mask_h_out[:, None] & mask_h_in[None, :], other=0.0)
            acc += tl.sum(w_vals * y_vals[None, :], axis=1)
        # add bias
        bias_vals = tl.load(Bias_ptr + h_out_idx, mask=mask_h_out, other=0.0)
        acc += bias_vals
        # store to Out[b, s, h_out_idx]
        out_offset = b * out_bs_stride + s * out_ss_stride + h_out_idx * out_hs_stride
        tl.store(Out_ptr + out_offset, acc, mask=mask_h_out)


class ModelNew(torch.nn.Module):
    def forward(self,
                x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        All heavy ops are implemented in Triton kernels; no PyTorch F.linear/F.conv1d in forward.
        """
        # Ensure contiguous tensors
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        I = 3 * H
        K = 4

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, I), dtype=x.dtype, device=x.device)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B, C, x_proj
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # 3) Left gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        total = B * S * H
        grid_gm = (total,)
        gated_mul_kernel_left[grid_gm](
            B_tensor, x_proj_tensor, Bx,
            B, S, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj_tensor.stride(0), x_proj_tensor.stride(1), x_proj_tensor.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4,
        )

        # 4) Pad Bx with causal left pad (K-1 zeros)
        Bx_padded = torch.empty((B, H, S + K - 1), dtype=x.dtype, device=x.device)
        grid_pad = (B * H, S + K - 1)
        pad1d_causal_kernel[grid_pad](
            Bx, Bx_padded,
            B, H, S, K,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            num_warps=4,
        )

        # 5) Grouped causal conv: conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=H)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        grid_conv = (B * H, 8)  # tile over S with BLOCK_T=256, 256//32=8 for num warps
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight.view(-1), conv_bias, conv_out,
            B, H, S, K,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            conv_weight.stride(0) if conv_weight.dim() == 3 else 0,
            conv_weight.stride(2) if conv_weight.dim() == 3 else 0,
            num_warps=4,
        )

        # 6) Right gating: y = C * conv_out
        y = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid_gm2 = (total,)
        gated_mul_kernel_right[grid_gm2](
            C_tensor.transpose(1, 2).contiguous(), conv_out.transpose(1, 2).contiguous(), y,
            B, S, H,
            C_tensor.transpose(1, 2).contiguous().stride(0), C_tensor.transpose(1, 2).contiguous().stride(1), C_tensor.transpose(1, 2).contiguous().stride(2),
            conv_out.transpose(1, 2).contiguous().stride(0), conv_out.transpose(1, 2).contiguous().stride(1), conv_out.transpose(1, 2).contiguous().stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4,
        )

        # 7) out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
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
