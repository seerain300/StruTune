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
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointers for this (b, s)
    x_base = X_ptr + b * x_b_stride + s * x_s_stride

    # For each output channel i in [0, I)
    # Accumulate in float32 and cast on store
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)

        # Reduce over H dimension
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            # Load weight row for output channel i
            w_vals = tl.load(W_ptr + i * W_ptr.stride(0) + h_offsets * W_ptr.stride(1), mask=h_mask, other=0.0)
            # Multiply-accumulate
            acc += tl.sum(x_vals * w_vals, axis=0)

        # Compute output pointer for this (b, s, i)
        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + i * out_i_stride
        # Store, casting to Out_ptr dtype (Triton infers element type from Out_ptr)
        # If Out_ptr is fp32, this is fine; otherwise Triton will cast as needed.
        tl.store(out_ptr, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr,         # *const float, input after causal padding: (B, H, S+pad)
    W_ptr,         # *const float, conv_weight: (H, 1, K), K=4
    Bias_ptr,      # *const float, conv_bias: (H)
    Out_ptr,       # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,   # original sequence length
    K: tl.int32,   # kernel size, e.g., 4
    pad: tl.int32, # causal padding, e.g., 1
    # strides
    x_b_stride: tl.int32, x_h_stride: tl.int32, x_s_stride: tl.int32,  # for X (B, H, S+pad)
    w_g_stride: tl.int32, w_k_stride: tl.int32,                          # for W (H, 1, K)
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,  # for Out (B, H, S)
    BLOCK_T: tl.constexpr,  # tile size along output sequence length
):
    # Grid: axis=0 over B*H (one program per (b, g))
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # Output length
    T_out = S + pad

    # Base pointers for this (b, g)
    x_base = X_ptr + b * x_b_stride + g * x_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # Iterate over output positions in tiles
    for t0 in range(0, T_out, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_out

        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Convolution with K=4, causal padding: t_in = t_offsets + k - pad
        # Only t_in >= 0 contributes
        # Manually unroll k loop for K=4
        # k = 0
        t_in0 = t_offsets + 0 - pad
        valid0 = t_mask & (t_in0 >= 0)
        x0 = tl.load(x_base + t_in0 * x_s_stride, mask=valid0, other=0.0)
        w0 = tl.load(W_ptr + g * w_g_stride + 0 * w_k_stride, mask=True, other=0.0)
        acc += x0 * w0

        # k = 1
        t_in1 = t_offsets + 1 - pad
        valid1 = t_mask & (t_in1 >= 0)
        x1 = tl.load(x_base + t_in1 * x_s_stride, mask=valid1, other=0.0)
        w1 = tl.load(W_ptr + g * w_g_stride + 1 * w_k_stride, mask=True, other=0.0)
        acc += x1 * w1

        # k = 2
        t_in2 = t_offsets + 2 - pad
        valid2 = t_mask & (t_in2 >= 0)
        x2 = tl.load(x_base + t_in2 * x_s_stride, mask=valid2, other=0.0)
        w2 = tl.load(W_ptr + g * w_g_stride + 2 * w_k_stride, mask=True, other=0.0)
        acc += x2 * w2

        # k = 3
        t_in3 = t_offsets + 3 - pad
        valid3 = t_mask & (t_in3 >= 0)
        x3 = tl.load(x_base + t_in3 * x_s_stride, mask=valid3, other=0.0)
        w3 = tl.load(W_ptr + g * w_g_stride + 3 * w_k_stride, mask=True, other=0.0)
        acc += x3 * w3

        # Add bias
        bval = tl.load(Bias_ptr + g, mask=True, other=0.0)
        acc += bval

        # Store results to Out[b, g, t_offsets]
        out_ptrs = out_base + t_offsets * out_s_stride
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_hout_stride: tl.constexpr,  # stride along H_out (row)
    w_hin_stride: tl.constexpr,   # stride along H_in (col)
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointers for this (b, s)
    y_base = Y_ptr + b * y_b_stride + s * y_s_stride

    # For each output channel h_out in [0, H)
    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over H_in dimension in tiles
        for h_in in range(0, H, BLOCK_H):
            h_offsets = h_in + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0)
            w_vals = tl.load(W_ptr + h_out * w_hout_stride + h_offsets * w_hin_stride, mask=h_mask, other=0.0)
            acc += tl.sum(y_vals * w_vals, axis=0)

        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + h_out * out_h_stride
        tl.store(out_ptr, acc)


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
        Triton-optimized fused computation:
        1) in_proj linear: BCx = F.linear(x, in_proj_weight, in_proj_bias)  -> (B, S, 3*H)
        2) Slice: B_tensor = BCx[:, :, :H], x_proj = BCx[:, :, 2*H:], C_tensor = BCx[:, :, H:2*H]
        3) Elementwise gating: Bx = B_tensor * x_proj  -> (B, S, H)
        4) Grouped causal conv on Bx: conv_out = conv(Bx, conv_weight, conv_bias, groups=H) -> (B, H, S)
        5) Output gating: y = C_tensor.transpose(-1, -2) * conv_out  -> (B, S, H)
        6) Final out projection: output = F.linear(y, out_proj_weight, out_proj_bias)
        """
        assert x.is_cuda and in_proj_weight.is_cuda and conv_weight.is_cuda and out_proj_weight.is_cuda, "All tensors must be CUDA for Triton."
        B, S, H = x.shape
        I = in_proj_weight.shape[0]  # 3*H
        K = conv_weight.shape[2]     # kernel size (4 in original)
        assert K == 4, "Only kernel_size=4 is supported in this Triton implementation."

        # 1) in_proj linear via Triton
        BCx_out = torch.empty((B, S, I), device=x.device, dtype=x.dtype)
        # Ensure x and weights are contiguous and cast compute to float32 in kernel, store to BCx_out dtype
        # Launch grid: (B*S,)
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x,
            in_proj_weight,
            BCx_out,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            BCx_out.stride(0), BCx_out.stride(1), BCx_out.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slicing
        B_tensor = BCx_out[:, :, :H]
        C_tensor = BCx_out[:, :, H:2 * H]
        x_proj = BCx_out[:, :, 2 * H:]

        # 3) Elementwise gating
        Bx = B_tensor * x_proj  # PyTorch elementwise multiply

        # 4) Grouped causal conv1d in Triton: operate on (B, H, S) input
        # Pad with causal padding on left
        pad = 1
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # (B, H, S+1)
        Bx_padded = Bx_padded.contiguous()
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        # Launch grid: (B*H,)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded,             # X (B, H, S+pad)
            conv_weight,           # W (H, 1, 4)
            conv_bias,             # Bias (H)
            conv_out,              # Out (B, H, S)
            B, H, S, K, pad,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256,
            num_warps=4,
        )

        # 5) Output gating: y = C_tensor.transpose(-1, -2) * conv_out
        # C_tensor is (B, S, H); conv_out is (B, H, S). We multiply C at position (b, s, h) with conv_out at (b, h, s).
        # To make shapes align, we can do:
        # First transpose C_tensor to (B, H, S) for elementwise multiply
        C_trans = C_tensor.transpose(-1, -2).contiguous()
        y = C_trans * conv_out  # elementwise multiply

        # 6) Final out projection via Triton
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y,                   # Y: (B, S, H)
            out_proj_weight,     # (H, H)
            output,              # (B, S, H)
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
