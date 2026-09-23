import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Elementwise gating: out = a * b, where a: (B, L, H), b: (B, L, H)
@triton.jit
def gate_mul_kernel(
    a_ptr, b_ptr, out_ptr,   # pointers
    B, L, H,                 # sizes
    stride_a_b, stride_a_l, stride_a_h,    # strides for a (B, L, H)
    stride_b_b, stride_b_l, stride_b_h,    # strides for b (B, L, H)
    stride_out_b, stride_out_l, stride_out_h   # strides for out (B, L, H)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    a_ptrs = a_ptr + b * stride_a_b + l_offsets[None, :] * stride_a_l + h_offsets[:, None] * stride_a_h
    b_ptrs = b_ptr + b * stride_b_b + l_offsets[None, :] * stride_b_l + h_offsets[:, None] * stride_b_h
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h

    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    out_vals = a_vals * b_vals
    tl.store(out_ptrs, out_vals, mask=mask)


# Grouped causal 1D convolution:
# Input Bx: (B, H, L_padded), where L_padded = L + K - 1 (K = 4), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L_padded), contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L_padded)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l # strides for out (B, H, L)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]
    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    # Accumulator for output tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # K = 4 (kernel_size), loop over kernel positions
    for k in range(0, 4):
        # For each output channel h, sum contributions from its input channel h and position k
        for ho in range(0, H):
            # Bx[b, ho, l + k] -> since Bx is padded on the right by K-1, we use l_offsets + k
            in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + (l_offsets[None, :] + k) * stride_Bx_l
            in_vals = tl.load(in_ptrs, mask=mask, other=0.0)  # (1, 128), broadcasts over ho

            # Weight w[ho, ho, k] (groups=H -> output_channel == input_channel)
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            acc += w_val * in_vals  # broadcast w_val to (64, 128)

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

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
        x: (B, L, H)
        in_proj_weight: (3H, H, L)
        in_proj_bias: (3H)
        conv_weight: (H, H, 4)
        conv_bias: (H)
        out_proj_weight: (H, L, H)  # original code's final linear weight
        out_proj_bias: (H)
        """
        B, L, H = x.shape
        device = x.device

        # 1) In-projection: BCx = linear(x, in_proj_weight, in_proj_bias) -> (B, 3H, L)
        # Note: in_proj_weight shape (3H, H, L), input x shape (B, L, H)
        BCx = F.linear(x, in_proj_weight, in_proj_bias)  # (B, 3H, L)

        # 2) Transpose for conv: (B, 3H, L) -> (B, L, 3H)
        BCx_T = BCx.transpose(-1, -2)

        # 3) Chunk into B_tensor, C_tensor, x_proj, each (B, H, L) using view/slice
        B_tensor = BCx_T[:, :, :H].transpose(-1, -2)  # (B, H, L)
        C_tensor = BCx_T[:, :, H:2*H].transpose(-1, -2)  # (B, H, L)
        x_proj = BCx_T[:, :, 2*H:].transpose(-1, -2)  # (B, H, L)

        # 4) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, H, L), device=device, dtype=torch.float32)
        grid_gm = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gm](
            B_tensor, x_proj, Bx,
            B, L, H,
            *B_tensor.stride(), *x_proj.stride(), *Bx.stride(),
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal 1D convolution with padding
        # Pad Bx on the last dimension by K-1 (K=4) to make it (B, H, L+3)
        L_padded = L + 3
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad 3 zeros on the left, causal

        # Output tensor for conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), device=device, dtype=torch.float32)

        # Launch Triton grouped causal conv kernel
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, L, H,
            *Bx_padded.stride(), *conv_weight.stride(), *conv_out.stride(),
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_tensor * conv_out
        y = torch.empty((B, H, L), device=device, dtype=torch.float32)
        grid_gm2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gm2](
            C_tensor, conv_out, y,
            B, L, H,
            *C_tensor.stride(), *conv_out.stride(), *y.stride(),
            num_warps=4, num_stages=2
        )

        # 7) Final out-projection: y (B, H, L) -> output (B, L, H) using F.linear
        # Note: out_proj_weight is (H, L, H), F.linear(y_T, out_proj_weight, bias) expects y_T (B, L, H)
        y_T = y.transpose(-1, -2)  # (B, H, L) -> (B, L, H)
        output = F.linear(y_T, out_proj_weight, out_proj_bias)  # (B, L, H)

        return output


def run(*args):
    return ModelNew()(*args)
