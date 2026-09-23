import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const T, input x: (B, S, H)
    W_ptr,         # *const T, in_proj_weight: (I, H), I=3*H
    BIAS_ptr,      # *const T, in_proj_bias: (I,) or None
    Out_ptr,       # *T, output: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for w
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    HAS_BIAS: tl.int32,  # 1 if bias provided, else 0
    BLOCK_H: tl.constexpr,
):
    # Grid: (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)

        # reduce over H dimension in tiles
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            )
            # ensure float32 for accumulation
            x_vals = x_vals.to(tl.float32)

            w_vals = tl.load(
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            )
            w_vals = w_vals.to(tl.float32)

            acc += tl.sum(x_vals * w_vals, axis=0)

        if HAS_BIAS:
            bias_val = tl.load(BIAS_ptr + i).to(tl.float32)
            acc += bias_val

        # store to output; Out_ptr element type defines dtype
        out_ptr_i = out_base + i * out_i_stride
        # Triton will cast to pointer element type on store
        tl.store(out_ptr_i, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr,         # *const float, input after linear and gating: (B, H, S)
    W_ptr,         # *const float, conv_weight: (H, 1, 4)
    BIAS_ptr,      # *const float, conv_bias: (H,)
    Out_ptr,       # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    K: tl.int32,   # kernel_size
    # strides for x (B, H, S)
    x_b_stride: tl.int32, x_h_stride: tl.int32, x_s_stride: tl.int32,
    # strides for w (H, 1, K)
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    # strides for out (B, H, S)
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # grid: (B, H) each program handles one (b, g)
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)
    b = pid_b
    g = pid_g

    x_base = X_ptr + b * x_b_stride + g * x_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # iterate over output positions t in tiles
    for t0 in range(0, S, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # causal conv with K=4 and padding K-1 on the left
        for k in range(0, K):
            # idx = t + k - (K-1)  # causal alignment
            idx = t_offsets + k - (K - 1)
            valid = (idx >= 0) & (idx < S) & t_mask

            x_vals = tl.load(
                x_base + idx * x_s_stride,
                mask=valid,
                other=0.0
            ).to(tl.float32)

            w_val = tl.load(
                W_ptr + g * w_g_stride + k * w_k_stride
            ).to(tl.float32)

            acc += x_vals * w_val

        if tl.load(BIAS_ptr + g).to(tl.float32) != 0.0:
            bias_val = tl.load(BIAS_ptr + g).to(tl.float32)
            acc += bias_val

        out_ptrs = out_base + t_offsets * out_s_stride
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    BIAS_ptr,      # *const float, out_proj_bias: (H,)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y (B, S, H)
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for w (H, H)
    w_out_h_out_stride: tl.int32, w_out_h_in_stride: tl.int32,
    # strides for out (B, S, H)
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid: (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)

        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            w_vals = tl.load(
                W_ptr + h_out * w_out_h_out_stride + h_in_offsets * w_out_h_in_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)

            acc += tl.sum(y_vals * w_vals, axis=0)

        bias_val = tl.load(BIAS_ptr + h_out).to(tl.float32)
        acc += bias_val

        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        """
        Triton-optimized forward that performs:
        1) in_proj linear: x -> (B, S, I) via in_proj_weight, in_proj_bias
        2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        3) Elementwise Bx = B_tensor * x_proj_tensor
        4) Grouped causal conv on Bx with kernel_size=4, groups=H
        5) y = C_tensor * conv_out
        6) out_proj: y -> (B, S, H) via out_proj_weight, out_proj_bias
        """
        assert x.dim() == 3, "x must be (B, S, H)"
        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj linear: (B, S, H) -> (B, S, I)
        x_ = x.contiguous()
        w_in = in_proj_weight.contiguous()
        if in_proj_bias is not None:
            bias_in = in_proj_bias.contiguous()
        else:
            bias_in = torch.zeros(w_in.shape[0], dtype=x_.dtype, device=x_.device)
        out_bc = torch.empty((B, S, I), dtype=x_.dtype, device=x_.device)

        in_proj_linear_kernel[(B * S,)](
            x_, w_in, bias_in, out_bc,
            B, S, H, I,
            x_.stride(0), x_.stride(1), x_.stride(2),
            w_in.stride(0), w_in.stride(1),
            out_bc.stride(0), out_bc.stride(1), out_bc.stride(2),
            int(bias_in is not None),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj_tensor
        # BCx shape: (B, S, 3*H)
        BCx = out_bc  # (B, S, I)
        # Slice: B: [0:H), C: [H:2*H), x_proj: [2*H:3*H)
        B_tensor = BCx[:, :, :H]           # (B, S, H)
        x_proj_tensor = BCx[:, :, 2 * H:]  # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]      # (B, S, H)

        # 3) Elementwise Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # Transpose for conv: (B, H, S)
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, S, H) -> (B, H, S)

        # 4) Grouped causal conv with K=4, groups=H
        # conv_weight: (H, 1, 4), conv_bias: (H,)
        conv_w = conv_weight.contiguous()
        conv_b = conv_bias.contiguous()
        conv_out = torch.empty((B, H, S), dtype=x_.dtype, device=x_.device)

        grouped_causal_conv1d_kernel[(B, H)](
            Bx_trans, conv_w, conv_b, conv_out,
            B, H, S, 4,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_w.stride(0), conv_w.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=256,
            num_warps=4,
        )

        # 5) y = C_tensor * conv_out  ; conv_out is (B, H, S)
        y = C_tensor * conv_out  # (B, S, H)

        # Transpose back to (B, H, S) for conv semantics, then elementwise:
        # y already (B, S, H)

        # 6) out_proj linear: (B, S, H) -> (B, S, H)
        y_ = y.contiguous()
        w_out = out_proj_weight.contiguous()
        bias_out = out_proj_bias.contiguous()
        out = torch.empty((B, S, H), dtype=x_.dtype, device=x_.device)

        out_proj_linear_kernel[(B * S,)](
            y_, w_out, bias_out, out,
            B, S, H,
            y_.stride(0), y_.stride(1), y_.stride(2),
            w_out.stride(0), w_out.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
