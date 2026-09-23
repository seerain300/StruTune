import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: linear projection Y[b, s, h] = sum_i X[b, s, i] * W[h, i] + bias[h]
# Inputs:
#   X: [B, S, H], W: [H, H], bias: [H], Y: [B, S, H]
@triton.jit
def TritonLinearProjKernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # Accumulator for each (s, h) tile
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Loop over input feature dimension H (reduction over i)
    # X[b, s, i] * W[h, i] + bias[h]
    # We iterate i from 0 to H-1 and accumulate into acc[h] per s.
    # Note: Since H is typically not a multiple of BLOCK_H, we accumulate row-wise.
    # To keep simplicity and correctness, we handle reduction by looping i.
    for i in range(0, H):
        # Load X tile: shape (BLOCK_S, 1)
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + i * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)

        # Load W vector for all h in tile: shape (1, BLOCK_H)
        w_ptrs = W_ptr + h_offsets[None, :] * (H) + i
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)

        # Multiply-accumulate: (BLOCK_S, 1) * (1, BLOCK_H) -> (BLOCK_S, BLOCK_H)
        acc += x_vals * w_vals

    # Add bias
    bias_ptrs = bias_ptr + h_offsets
    bias_vals = tl.load(bias_ptrs, mask=mask_h, other=0.0)
    acc += bias_vals[None, :]  # broadcast over s-dimension

    # Store results
    y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    stride_b_b, stride_b_s, stride_b_h,
    stride_x_b, stride_x_s, stride_x_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
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


# Triton kernel: left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
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


# Triton kernel: grouped causal 1D convolution
# Input Bx_pad [B, H, S+3], weight conv_weight [H, 1, 4], bias conv_bias [H]
# Output conv_out [B, H, S] where conv_out[b, h, s] = sum_{k=0..3} Bx_pad[b, h, s + k] * conv_weight[h, 0, k] + conv_bias[h]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_w_ptr, conv_b_ptr, conv_out_ptr,
    B, H, S,
    stride_bp_b, stride_bp_h, stride_bp_s,
    stride_w_h, stride_w_k,
    stride_co_b, stride_co_h, stride_co_s,
    PAD: tl.constexpr,  # PAD=3
    K: tl.constexpr,    # K=4
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Convolution over k=0..3
    for k in range(0, K):
        # Read Bx_pad[b, h, s + k]
        src_s = s_offsets + k  # padded indices
        # Ensure src_s is within [PAD, S+PAD-1]
        valid = (src_s >= PAD) & (src_s < (S + PAD)) & mask_s
        bx_ptrs = Bx_pad_ptr + b * stride_bp_b + h * stride_bp_h + src_s * stride_bp_s
        bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0)

        # Read conv_weight[h, 0, k]
        w_ptr = conv_w_ptr + h * stride_w_h + k * stride_w_k
        w_val = tl.load(w_ptr)  # scalar
        acc += bx_vals * w_val

    # Add bias
    b_ptr = conv_b_ptr + h
    b_val = tl.load(b_ptr)
    acc += b_val

    # Store conv_out[b, h, s]
    co_ptrs = conv_out_ptr + b * stride_co_b + h * stride_co_h + s_offsets * stride_co_s
    tl.store(co_ptrs, acc, mask=mask_s)


# Triton kernel: final output projection over (B, S, H)
# Y [B, S, H] = C * conv_out [B, H, S] (we pass Y_transposed as [B, S, H])
# out[b, s, h] = sum_j y[b, s, j] * out_proj_weight[h, j] + out_proj_bias[h]
@triton.jit
def TritonFinalProjKernel(
    Y_ptr, out_proj_w_ptr, out_proj_b_ptr, out_ptr,
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_h, stride_w_j,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Reduction over j (features) to produce (B, S, H)
    for j in range(0, H):
        # Load Y[b, s, j] as (BLOCK_S, 1)
        y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + j * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0)

        # Load out_proj_weight[h, j] as (1, BLOCK_H)
        w_ptrs = out_proj_w_ptr + h_offsets[None, :] * stride_w_h + j * stride_w_j
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)

        acc += y_vals * w_vals

    # Add bias
    b_ptrs = out_proj_b_ptr + h_offsets
    b_vals = tl.load(b_ptrs, mask=mask_h, other=0.0)
    acc += b_vals[None, :]

    # Store out[b, s, h]
    out_ptrs = out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward that mirrors the original computation:
        - Three linear projections (B, C, x_proj) using TritonLinearProjKernel
        - Element-wise gating (Bx = B * x_proj) using TritonGateKernel
        - Left-pad for causal conv using TritonPadLeftKernel
        - Grouped causal conv (H, 1, 4), groups=H, using TritonGroupedCausalConvKernel
        - Output gating y = C * conv_out (elementwise in Triton)
        - Final output projection using TritonFinalProjKernel
        """
        # Ensure CUDA tensors and contiguity
        assert x.is_cuda, "Input x must be on CUDA device."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()  # (3H, H)
        in_proj_bias = in_proj_bias.contiguous()      # (3H)
        conv_weight = conv_weight.contiguous()        # (H, 1, 4)
        conv_bias = conv_bias.contiguous()            # (H)
        out_proj_weight = out_proj_weight.contiguous()  # (H, H)
        out_proj_bias = out_proj_bias.contiguous()      # (H)

        B, S, H = x.shape

        # 1) Three linear projections: each produces (B, S, H)
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

        # 2) Element-wise gating with Triton
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            B_lin, x_proj_lin, Bx,
            B, S, H,
            B_lin.stride(0), B_lin.stride(1), B_lin.stride(2),
            x_proj_lin.stride(0), x_proj_lin.stride(1), x_proj_lin.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # 3) Left-pad along sequence by 3 for causal conv
        # Bx_pad shape: (B, H, S + 3)
        S_padded = S + 3
        Bx_pad = torch.empty((B, H, S_padded), dtype=x.dtype, device=x.device)
        TritonPadLeftKernel[(B, H, triton.cdiv(S_padded, 128))](
            Bx, Bx_pad,
            B, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=128,
        )

        # 4) Grouped causal conv: conv_out [B, H, S]
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        TritonGroupedCausalConvKernel[(B, H, triton.cdiv(S, 64))](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, S,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),  # stride for k is 1, but we pass stride_w_k explicitly
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            PAD=3, K=4, BLOCK_S=64,
        )

        # 5) Output gating: y = C * conv_out, shape (B, H, S)
        y = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, H, triton.cdiv(S, 64))](
            C_lin, conv_out, y,
            B, H, S,
            C_lin.stride(0), C_lin.stride(1), C_lin.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=64, BLOCK_H=64,
        )

        # 6) Final output projection: produce (B, S, H)
        out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonFinalProjKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        return out


def run(*args):
    return ModelNew()(*args)
