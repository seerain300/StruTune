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
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # grid over (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # base pointers for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            h_offsets = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            # load X[b, s, h_offsets]
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0).to(tl.float32)

            # load W[i, h_offsets] (W is (I, H))
            w_vals = tl.load(W_ptr + i * W_ptr.shape[1] + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

            # reduce
            acc += tl.sum(x_vals * w_vals, axis=0)

        # store result at Out[b, s, i]
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    In_ptr,        # *const float, input after gating: (B, H, S)
    W_ptr,         # *const float, conv_weight: (H, 1, 4)
    BIAS_ptr,      # *const float, conv_bias: (H)
    Out_ptr,       # *float, conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    K: tl.int32,   # kernel_size (here 4)
    # strides for input
    in_b_stride: tl.int32, in_g_stride: tl.int32, in_t_stride: tl.int32,
    # strides for output
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,  # tile size along S
):
    # grid over (B*H,)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # base pointers for this (b, g)
    in_base = In_ptr + b * in_b_stride + g * in_g_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # output positions t
    t_start = 0
    while t_start < S:
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        # accumulate over kernel k in [0..K-1]
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # causal padding: effective input indices are t + k - 1
        for k in range(0, K):
            t_eff = t_offsets + (k - 1)
            mask_eff = (t_eff >= 0) & (t_eff < S) & t_mask

            # load input values: In[b, g, t_eff]
            in_ptrs = in_base + t_eff * in_t_stride
            x = tl.load(in_ptrs, mask=mask_eff, other=0.0).to(tl.float32)

            # load conv weight for group g and kernel k: W[g, 0, k]
            # W shape: (H, 1, 4)
            w_val = tl.load(W_ptr + g * W_ptr.shape[0] * W_ptr.shape[2] + 0 * W_ptr.shape[1] + k * W_ptr.shape[2]).to(tl.float32)

            acc += x * w_val

        # add bias
        bias_val = tl.load(BIAS_ptr + g).to(tl.float32)
        acc += bias_val

        # store conv_out[b, g, t_offsets]
        out_ptrs = out_base + t_offsets * out_t_stride
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_out_ptr,     # *const float, out_proj_weight: (H, H)
    Bias_out_ptr,  # *const float, out_proj_bias: (H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for Y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile over H
):
    # grid over (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            h_offsets = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_out_ptr + h_out * W_out_ptr.shape[1] + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        # add bias
        bval = tl.load(Bias_out_ptr + h_out).to(tl.float32)
        acc += bval

        # store
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized fused computation:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3*H)
        2) Slice: B_tensor = BCx[:, :, :H], C_tensor = BCx[:, :, H:2*H], x_proj_tensor = BCx[:, :, 2*H:]
        3) Gating: Bx = B_tensor * x_proj_tensor
        4) Conv: conv_out = causal conv1d(Bx.transpose(-1, -2) with kernel_size=4, groups=H)
        5) Gating: y = C_tensor * conv_out.transpose(-1, -2)
        6) out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        """
        # Ensure tensors are contiguous and float32 for compute
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else None
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else None
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else None

        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj linear: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)

        # Launch Triton in_proj kernel
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64, num_warps=4, num_stages=2
        )

        # 2) Slice BCx into B_tensor, C_tensor, x_proj_tensor
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2*H]
        x_proj_tensor = BCx[:, :, 2*H:]

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 4) Prepare for conv: Bx_transposed (B, H, S) for conv1d
        Bx_trans = Bx.transpose(-1, -2).contiguous()  # (B, H, S)

        # Conv: groups=H, kernel_size=4, causal padding = K-1 = 3
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight, conv_bias,
            conv_out,
            B, H, S, 4,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256, num_warps=4, num_stages=2
        )

        # 5) Gating with C_tensor: y = C_tensor * conv_out (shape (B, H, S))
        # Then transpose back to (B, S, H)
        y = (C_tensor.transpose(-1, -2) * conv_out).contiguous()  # (B, S, H)

        # 6) out_proj: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias,
            output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
