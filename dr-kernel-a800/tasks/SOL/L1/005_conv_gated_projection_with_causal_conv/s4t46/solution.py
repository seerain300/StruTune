import torch
import triton
import triton.language as tl

@triton.jit
def in_proj_linear_kernel(
    X_ptr,          # *const T, input x: (B, S, H)
    W_ptr,          # *const T, in_proj_weight: (I, H), I=3*H
    Bias_ptr,       # *const T, in_proj_bias: (I)
    Out_ptr,        # *T, output BCx: (B, S, I)
    B: tl.int32,    # batch size
    S: tl.int32,    # sequence length
    H: tl.int32,    # hidden size
    I: tl.int32,    # output channels = 3*H
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    grid = tl.num_programs(axis=0)
    b = tl.program_id(axis=0) // S
    s = tl.program_id(axis=0) % S

    # Base pointers
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # Iterate over output channels I
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over H
        for h0 in range(0, H, BLOCK_H):
            h_offsets = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(W_ptr + i * W_ptr.shape[1] + h_offsets, mask=h_mask, other=0.0).to(tl.float32)  # W[i, h]
            acc += tl.sum(x_vals * w_vals, axis=0)
        # add bias[i]
        bias_i = tl.load(Bias_ptr + i).to(tl.float32)
        acc += bias_i

        # Store to Out[b, s, i] with casting to output dtype
        out_ptr_i = out_base + i * out_i_stride
        # Triton requires value's dtype to match pointer's element type on store; we cast here
        tl.store(out_ptr_i, acc.to(Out_ptr.dtype.element_ty))

@triton.jit
def grouped_causal_conv1d_kernel(
    Xpad_ptr,       # *const T, padded input Bx_padded: (B, H, S+pad)
    W_ptr,          # *const T, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const T, conv_bias: (H)
    Out_ptr,        # *T, conv_out: (B, H, S)
    B: tl.int32,    # batch size
    H: tl.int32,    # hidden size (groups)
    S: tl.int32,    # original seq_len
    K: tl.int32,    # kernel_size (here 4)
    pad: tl.int32,  # causal pad (K-1)
    # strides
    xpad_b_stride: tl.int32, xpad_g_stride: tl.int32, xpad_t_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_g_stride: tl.int32, out_t_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # axis=0 over B*H groups
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # base pointers for this (b, g)
    xpad_base = Xpad_ptr + b * xpad_b_stride + g * xpad_g_stride
    out_base = Out_ptr + b * out_b_stride + g * out_g_stride

    # Vector of output positions
    T_out = S
    for t0 in range(0, T_out, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_out

        # Initialize accumulator
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # conv with K=4, causal: input index is t + k - 1
        # Loop over k
        for k in range(0, K):
            in_t = t_offsets + k - pad  # left-pad by pad
            valid = (in_t >= 0) & (in_t < S) & t_mask
            x_ptrs = xpad_base + in_t * xpad_t_stride
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
            w_k = tl.load(W_ptr + g * w_g_stride + k * w_k_stride).to(tl.float32)  # scalar
            acc += x_vals * w_k

        # add bias[g]
        bias_g = tl.load(Bias_ptr + g).to(tl.float32)
        acc += bias_g

        # store to conv_out[b, g, t_offsets]
        out_ptrs = out_base + t_offsets * out_t_stride
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=t_mask)

@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const T, input y: (B, S, H)
    Wout_ptr,       # *const T, out_proj_weight: (H, H)
    Bout_bias_ptr,  # *const T, out_proj_bias: (H)
    Out_ptr,        # *T, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    wout_h_out_stride: tl.int32, wout_h_in_stride: tl.int32,
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
        for h_in0 in range(0, H, BLOCK_H):
            h_in_offsets = h_in0 + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=h_in_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(Wout_ptr + h_out * wout_h_out_stride + h_in_offsets * wout_h_in_stride, mask=h_in_mask, other=0.0).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        # add bias
        bias_h = tl.load(Bout_bias_ptr + h_out).to(tl.float32)
        acc += bias_h
        tl.store(out_base + h_out * out_h_stride, acc.to(Out_ptr.dtype.element_ty))

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
        # Ensure contiguous tensors
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        I = 3 * H  # in_proj output channels
        K = conv_weight.shape[2]  # kernel_size
        assert conv_weight.shape == (H, 1, K), "conv_weight must have shape (H, 1, 4)"
        assert out_proj_weight.shape[1] == H and out_proj_weight.shape[0] == H, "out_proj_weight must be (H, H)"

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=x.dtype)
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        # Shapes: (B, H, S)
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2*H]
        x_proj_tensor = BCx[:, :, 2*H:]

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, H, S)

        # 4) Causal padding for conv: pad = K - 1
        pad = K - 1
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # pad left by pad, no right pad
        Bx_padded = Bx_padded.contiguous()  # (B, H, S+pad)

        # 5) Grouped causal conv: conv_out = conv(Bx_padded, conv_weight, conv_bias, groups=H) -> (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, H, S, K, pad,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=128,
            num_warps=4,
        )

        # 6) Output gating: y = C_tensor.transpose(-1, -2) * conv_out
        # Shapes: C_tensor (B, H, S), conv_out (B, H, S) -> elementwise multiply -> (B, H, S)
        C_T = C_tensor.transpose(-1, -2).contiguous()  # (B, S, H)
        y = C_T * conv_out  # (B, S, H)

        # 7) Out projection: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
