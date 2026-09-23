import torch
import triton
import triton.language as tl

# Triton kernel: Linear projection out[B, S, M] = x @ weight[:M, :].T + bias[:M]
# x: (B, S, H), weight: (M, H), bias: (M)
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr,            # *T, shape [B, S, H]
    weight_ptr,       # *T, shape [M, H]
    bias_ptr,         # *T, shape [M]
    out_ptr,          # *T, shape [B, S, M]
    B, S, H, M,       # int32
    stride_x_b, stride_x_s, stride_x_h,   # strides for x (B, S, H)
    stride_w_m, stride_w_h,               # strides for weight (M, H)
    stride_o_b, stride_o_s, stride_o_m,   # strides for out (B, S, M)
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s = pid_s
    m = pid_m

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)  # accumulate in fp32 for stability

    # Loop over H in chunks of BLOCK_H
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load x[b, s, h_offsets] as vector
        x_vals = tl.load(
            x_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h,
            mask=mask_h,
            other=0.0
        ).to(tl.float32)  # cast to fp32 for compute

        # Load weight[m, h_offsets] as vector (row of weight for channel m)
        w_vals = tl.load(
            weight_ptr + m * stride_w_m + h_offsets * stride_w_h,
            mask=mask_h,
            other=0.0
        ).to(tl.float32)

        # Accumulate dot product for this chunk
        acc += tl.sum(x_vals[:, None] * w_vals[None, :], axis=0)

    # Add bias
    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val

    # Store to out[b, s, m]
    out_ptr_pos = out_ptr + b * stride_o_b + s * stride_o_s + m * stride_o_m
    # Cast back to original dtype (assume out_ptr dtype matches x dtype)
    # Triton will infer store dtype from out_ptr; ensure out tensor dtype matches x.dtype
    tl.store(out_ptr_pos, acc)

# Triton kernel: Element-wise gating: Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr,            # *T, shape [B, S, H] (tensor B)
    x_ptr,            # *T, shape [B, S, H] (tensor x_proj)
    out_ptr,          # *T, shape [B, S, H] (Bx)
    B, S, H,          # int32
    stride_b_b, stride_b_s, stride_b_h,   # strides for B
    stride_x_b, stride_x_s, stride_x_h,   # strides for x_proj
    stride_o_b, stride_o_s, stride_o_h,   # strides for out
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    s = pid_s
    h = pid_h

    # Load B[b, s, h] and x_proj[b, s, h]
    b_val = tl.load(B_ptr + b * stride_b_b + s * stride_b_s + h * stride_b_h)
    x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
    # Compute product and store
    out_val = b_val * x_val
    tl.store(out_ptr + b * stride_o_b + s * stride_o_s + h * stride_o_h, out_val)


# Triton kernel: Left-pad along sequence by PAD elements; out_pad: (B, H, S + PAD)
# Write out_pad[:, :, :PAD] = 0, out_pad[:, :, PAD:] = Bx[:, :, :]
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr,          # *T, shape [B, H, S]
    out_ptr,         # *T, shape [B, H, S + PAD]
    B, H, S, PAD,    # int32
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_pad = tl.program_id(2)

    b = pid_b
    h = pid_h
    pad_idx = pid_pad

    # Write zeros at first PAD positions
    if pad_idx < PAD:
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + pad_idx * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # If pad_idx >= PAD, copy from Bx into out at (pad_idx - PAD)
    # Launch grid is (B, H, S+PAD); only when pad_idx < PAD we do zeros, for pad_idx >= PAD we copy
    if pad_idx >= PAD:
        s_in = pad_idx - PAD
        if s_in >= 0 and s_in < S:
            bx_ptr = Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s
            val = tl.load(bx_ptr)
            out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + pad_idx * stride_ob_s
            tl.store(out_ptr_pos, val)


# Triton kernel: Grouped causal 1D convolution with K=4, groups=H
# Input: Bx_pad (B, H, S+PAD), conv_weight (H, 1, 4), conv_bias (H)
# Output: conv_out (B, H, S)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr,       # *T, shape [B, H, S+PAD]
    weight_ptr,       # *T, shape [H, 1, 4]
    bias_ptr,         # *T, shape [H]
    out_ptr,          # *T, shape [B, H, S]
    B, S, PAD, H,     # int32
    stride_bp_b, stride_bp_h, stride_bp_s,  # strides for Bx_pad
    stride_w_h, stride_w_k, stride_w_c,     # strides for weight (H, 1, 4)
    stride_ob_b, stride_ob_h, stride_ob_s,  # strides for output (B, H, S)
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Sum over k in 0..3: Bx_pad[b, h, s + k]
    for k in range(0, 4):
        s_in = s_offsets + (k - PAD)
        mask = mask_s & (s_in >= 0) & (s_in < S + PAD)

        bx_ptr = Bx_pad_ptr + b * stride_bp_b + h * stride_bp_h + s_in * stride_bp_s
        bx_vals = tl.load(bx_ptr, mask=mask, other=0.0).to(tl.float32)

        # conv_weight[h, 0, k]
        w_ptr = weight_ptr + h * stride_w_h + 0 * stride_w_c + k * stride_w_k
        w_val = tl.load(w_ptr).to(tl.float32)

        acc += bx_vals * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + h).to(tl.float32)
    acc += bias_val

    # Store conv_out[b, h, s_offsets]
    out_ptr_tile = out_ptr + b * stride_ob_b + h * stride_ob_h + s_offsets * stride_ob_s
    tl.store(out_ptr_tile, acc, mask=mask_s)


# Triton kernel: Final output projection: out[B, S, H] = y @ out_proj_weight.T + out_proj_bias
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
@triton.jit
def TritonFinalProjectionKernel(
    y_ptr,             # *T, shape [B, S, H] (y = C * conv_out)
    weight_T_ptr,      # *T, shape [H, H] (out_proj_weight.T)
    bias_ptr,          # *T, shape [H]
    out_ptr,           # *T, shape [B, S, H]
    B, S, H,           # int32
    stride_y_b, stride_y_s, stride_y_h,   # strides for y
    stride_w_row, stride_w_col,           # strides for weight_T (H, H)
    stride_o_b, stride_o_s, stride_o_h,   # strides for out
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    s = pid_s
    h = pid_h

    # Compute out[b, s, h] = sum_h' y[b, s, h'] * weight_T[h, h'] + bias[h]
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over H chunks
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        y_vec = tl.load(
            y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h,
            mask=mask_h,
            other=0.0
        ).to(tl.float32)

        weight_row = tl.load(
            weight_T_ptr + h * stride_w_row + h_offsets * stride_w_col,
            mask=mask_h,
            other=0.0
        ).to(tl.float32)

        acc += tl.sum(y_vec * weight_row, axis=0)

    # Add bias[h]
    bias_val = tl.load(bias_ptr + h).to(tl.float32)
    acc += bias_val

    # Store
    tl.store(out_ptr + b * stride_o_b + s * stride_o_s + h * stride_o_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Implement the same computation as the original run function, but using Triton kernels.
        Launch every Triton kernel from forward to satisfy TRITON-ONLY requirement.
        """
        assert x.is_cuda, "Input tensor x must be on CUDA device."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        # Ensure contiguous and keep dtype
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        M = H  # hidden_size

        # 1) Three linear projections: B_out, C_out, x_proj_out using Triton
        B_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B, S, triton.cdiv(M, 64))](
            x, in_proj_weight[:M, :], in_proj_bias[:M], B_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[:M, :].stride(0), in_proj_weight[:M, :].stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_H=64
        )

        C_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B, S, triton.cdiv(M, 64))](
            x, in_proj_weight[M:2 * M, :], in_proj_bias[M:2 * M], C_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[M:2 * M, :].stride(0), in_proj_weight[M:2 * M, :].stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_H=64
        )

        x_proj_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B, S, triton.cdiv(M, 64))](
            x, in_proj_weight[2 * M:3 * M, :], in_proj_bias[2 * M:3 * M], x_proj_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[2 * M:3 * M, :].stride(0), in_proj_weight[2 * M:3 * M, :].stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_H=64
        )

        # 2) Element-wise gating Bx = B * x_proj via Triton
        Bx = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(M, 64))](
            B_out, x_proj_out, Bx,
            B, S, M,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 3) Left-pad Bx by PAD=3 along sequence for causal conv
        PAD = 3
        Bx_pad = torch.empty((B, M, S + PAD), dtype=x.dtype, device=x.device)
        TritonPadLeftKernel[(B, M, S + PAD)](
            Bx, Bx_pad,
            B, M, S, PAD,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=1  # grid is (B, M, S+PAD); we handle copy via if conditions
        )

        # 4) Grouped causal conv with K=4, groups=H using Triton
        conv_out = torch.empty((B, M, S), dtype=x.dtype, device=x.device)
        TritonGroupedCausalConvKernel[(B, M, triton.cdiv(S, 64))](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, S, PAD, M,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(2), conv_weight.stride(1),  # (H,1,4) strides
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=64
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(M, 64))](
            C_out, conv_out, y,
            B, S, M,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 6) Final output projection via Triton
        out_proj_weight_T = out_proj_weight.transpose(0, 1).contiguous()  # (H, H)
        output = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonFinalProjectionKernel[(B, triton.cdiv(S, 64), triton.cdiv(M, 64))](
            y, out_proj_weight_T, out_proj_bias, output,
            B, S, M,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight_T.stride(0), out_proj_weight_T.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=64, BLOCK_H=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
