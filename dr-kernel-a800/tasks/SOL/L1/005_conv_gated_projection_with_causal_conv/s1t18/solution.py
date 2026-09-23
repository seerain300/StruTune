import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Kernel: Linear projection Y[b, s, h] = sum_i x[b, s, i] * W[h, i] + bias[h]
# Input: X [B, S, H], W [H, H], Bias [H], Output Y [B, S, H]
@triton.jit
def TritonLinearProjKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Reduce over i in H to compute Y[b, s, h] = sum_i X[b, s, i] * W[h, i]
    for i in range(0, H):
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + i * stride_x_h
        w_vals = tl.load(W_ptr + h_offsets[None, :] * H + i, mask=mask_h[None, :], other=0.0)
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
        # acc += x_vals * w_vals
        acc += x_vals * w_vals

    # Add bias[h]
    bias_vals = tl.load(Bias_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals[None, :]

    # Store result
    y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Kernel: Element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    stride_b_b, stride_b_s, stride_b_h,
    stride_x_b, stride_x_s, stride_x_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * stride_b_b + s_offsets[:, None] * stride_b_s + h_offsets[None, :] * stride_b_h
    x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + h_offsets[None, :] * stride_x_h

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * stride_o_b + s_offsets[:, None] * stride_o_s + h_offsets[None, :] * stride_o_h
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Kernel: Left-pad along sequence by PAD=3 for Bx, producing Bx_pad [B, H, S+3]
# Input: Bx [B, H, S], Output: out_pad [B, H, S+3]
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # write zeros at the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            bx_ptr = Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s
            val = tl.load(bx_ptr)
            out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s
            tl.store(out_ptr_pos, val)


# Kernel: Grouped causal 1D convolution over Bx_pad (H channels, S length), K=4, groups=H.
# Input: Bx_pad [B, H, S+PAD], conv_weight [H, 1, 4], conv_bias [H]
# Output: conv_out [B, H, S]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_w_ptr, conv_b_ptr, conv_out_ptr,
    B, H, S, PAD,  # PAD=3
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_c, stride_w_k,
    stride_co_b, stride_co_h, stride_co_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_in = S + PAD

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # sum over k in {0..3}
    for k in range(0, 4):
        # index in padded sequence is s_offsets + k
        s_idx = s_offsets + k
        valid = s_idx < S_in
        bx_ptrs = Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + s_idx * stride_bx_s
        bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0)
        w_val = tl.load(conv_w_ptr + h * stride_w_h + 0 * stride_w_c + k * stride_w_k)
        acc += bx_vals * w_val

    # add bias[h]
    bias_val = tl.load(conv_b_ptr + h)
    acc += bias_val

    # store to conv_out[b, h, s_offsets]
    co_ptrs = conv_out_ptr + b * stride_co_b + h * stride_co_h + s_offsets * stride_co_s
    tl.store(co_ptrs, acc, mask=mask_s)


# Kernel: Final linear projection without using F.linear:
# We need to compute out[b, s, h] = sum_i conv_out[b, h, s] * out_proj_weight[h, i] + out_proj_bias[h]
# Note: conv_out is (B, H, S); out_proj_weight is (H, H). We implement this as:
#       acc[b, s, h] = sum_i conv_out[b, h, s] * out_proj_weight[h, i] + out_proj_bias[h]
# We will transpose out_proj_weight to (H, H) and load rows appropriately.
@triton.jit
def TritonFinalProjKernel(
    conv_out_ptr, out_proj_weight_ptr, out_proj_bias_ptr, out_ptr,
    B, H, S,
    stride_co_b, stride_co_h, stride_co_s,
    stride_ow_h, stride_ow_i,  # out_proj_weight is (H, H): stride_ow_h along rows, stride_ow_i along cols
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # For each output h, sum over i=0..H-1: conv_out[b, h, s] * out_proj_weight[h, i]
    # conv_out is (B, H, S); out_proj_weight is (H, H). We fix b, h, s in tiles and loop i.
    for i in range(0, H):
        co_ptrs = conv_out_ptr + pid_b * stride_co_b + h_offsets[None, :] * stride_co_h + s_offsets[:, None] * stride_co_s
        co_vals = tl.load(co_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
        ow_vals = tl.load(out_proj_weight_ptr + h_offsets[None, :] * stride_ow_h + i * stride_ow_i, mask=mask_h[None, :], other=0.0)
        acc += co_vals * ow_vals

    # Add bias[h]
    bias_vals = tl.load(out_proj_bias_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals[None, :]

    # Store to out[b, s, h]
    out_ptrs = out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward that mirrors the original computation:
        - Three linear projections using Triton (each produces (B, S, H))
        - Element-wise gating (Bx = B * x_proj) using Triton
        - Left-pad for causal conv using Triton
        - Grouped causal conv1d implemented in Triton (K=4, groups=H)
        - Final linear projection implemented in Triton
        """
        # Ensure CUDA tensors and contiguity
        assert x.is_cuda, "Input x must be on CUDA device."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape

        # 1) Three linear projections using Triton
        B_lin = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearProjKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_lin,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            B_lin.stride(0), B_lin.stride(1), B_lin.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        C_lin = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearProjKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H], C_lin,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            C_lin.stride(0), C_lin.stride(1), C_lin.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        x_proj_lin = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearProjKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H], x_proj_lin,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            x_proj_lin.stride(0), x_proj_lin.stride(1), x_proj_lin.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # 2) Element-wise gating with Triton: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            B_lin, x_proj_lin, Bx,
            B, S, H,
            B_lin.stride(0), B_lin.stride(1), B_lin.stride(2),
            x_proj_lin.stride(0), x_proj_lin.stride(1), x_proj_lin.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # 3) Left-pad along sequence by 3 using Triton
        PAD = 3
        S_in = S + PAD
        Bx_pad = torch.empty((B, H, S_in), dtype=Bx.dtype, device=Bx.device)
        TritonPadLeftKernel[(B, H, triton.cdiv(S_in, 128))](
            Bx, Bx_pad,
            B, H, S, PAD,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=128,
        )

        # 4) Grouped causal conv implemented in Triton (K=4, groups=H)
        conv_out = torch.empty((B, H, S), dtype=Bx_pad.dtype, device=Bx_pad.device)
        TritonGroupedCausalConvKernel[(B, H, triton.cdiv(S, 128))](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, S, PAD,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
        )

        # 5) Final output projection implemented in Triton: out[b, s, h] = sum_i conv_out[b, h, s] * out_proj_weight[h, i] + out_proj_bias[h]
        out = torch.empty((B, S, H), dtype=conv_out.dtype, device=conv_out.device)
        TritonFinalProjKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            conv_out, out_proj_weight, out_proj_bias, out,
            B, H, S,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),  # out_proj_weight is (H, H)
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        return out


def run(*args):
    return ModelNew()(*args)
