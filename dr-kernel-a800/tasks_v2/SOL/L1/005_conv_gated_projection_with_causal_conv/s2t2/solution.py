import torch
import triton
import triton.language as tl


# Kernel 1: Single Linear Projection (generic)
# Computes out[b, s, h] = sum over k of x[b, s, k] * w[h, k] + bias[h]
@triton.jit
def linear_bsh_kernel(
    x_ptr,         # *float32, input tensor x of shape (B, S, H)
    w_ptr,         # *float32, weight matrix of shape (H, H)
    b_ptr,         # *float32, bias vector of shape (H,)
    out_ptr,       # *float32, output tensor of shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,
    stride_wk, stride_wh,          # w is (H, H): k is column, h is row
    stride_outb, stride_outs, stride_outh,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = 0.0
    for k in range(0, H):
        x_val = tl.load(x_ptr + b * stride_xb + s * stride_xs + k * stride_xh)
        w_val = tl.load(w_ptr + h_out * stride_wh + k * stride_wk)
        acc += x_val * w_val
    out_val = acc + tl.load(b_ptr + h_out * stride_wh)  # bias index by h_out
    tl.store(out_ptr + b * stride_outb + s * stride_outs + h_out * stride_outh, out_val)


# Kernel 2: Grouped Causal 1D Convolution (depthwise, kernel_size=4)
# Input Bx: (B, H, S) contiguous (we assume Bx is passed by the caller or handled outside forward)
# conv_weight: (H, H, 4) contiguous
# conv_bias: (H,) contiguous
# Output conv_out: (B, H, S) contiguous (we write into conv_out provided)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *float32, (B, H, S)
    convW_ptr,     # *float32, (H, H, 4)
    convB_ptr,     # *float32, (H,)
    convO_ptr,     # *float32, (B, H, S)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_Bxb, stride_Bxh, stride_Bxs,
    stride_Wc, stride_Wh, stride_Wk,
    stride_Ob, stride_Oh, stride_Os,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)  # output position

    acc = 0.0
    # kernel_size = 4, causal padding: index t + k - 1
    for k in range(4):
        in_pos = t + k - 1
        in_valid = (in_pos >= 0) & (in_pos < S)
        # Load Bx[b, c, in_pos]
        bx_ptr = Bx_ptr + b * stride_Bxb + c * stride_Bxh + in_pos * stride_Bxs
        bx_val = 0.0
        if in_valid:
            bx_val = tl.load(bx_ptr)
        # Load conv_weight[c, c, k]
        w_ptr = convW_ptr + c * stride_Wc + c * stride_Wh + k * stride_Wk
        w_val = tl.load(w_ptr)
        acc += bx_val * w_val

    bias_val = tl.load(convB_ptr + c * stride_Oh)
    out_ptr = convO_ptr + b * stride_Ob + c * stride_Oh + t * stride_Os
    tl.store(out_ptr, acc + bias_val)


# Kernel 3: Final Linear (GEMV-style over H), output shape (B, S, H)
# Input y: (B, S, H) contiguous
# out_proj_weight: (H, H) contiguous
# out_proj_bias: (H,) contiguous
# Output out: (B, S, H) contiguous
@triton.jit
def final_linear_gemv_bsh_kernel(
    y_ptr,           # *float32, (B, S, H)
    wy_ptr,          # *float32, (H, H)
    bb_ptr,          # *float32, (H,)
    out_ptr,         # *float32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_yb, stride_ys, stride_yh,
    stride_wyn, stride_wyk,        # wy is (H,H): n=channel (output feature), k=input feature
    stride_ob, stride_os, stride_oh,
    K_BLOCK: tl.constexpr,         # tile size over K (H)
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    for h_out in range(0, H):
        acc = 0.0
        for k in range(0, H, K_BLOCK):
            k_offsets = k + tl.arange(0, K_BLOCK)
            mask_k = k_offsets < H
            # y[b, s, k_offsets]
            y_ptrs = y_ptr + b * stride_yb + s * stride_ys + k_offsets * stride_yh
            y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # (K_BLOCK,)
            # wy[h_out, k_offsets] -> (K_BLOCK,)
            wy_ptrs = wy_ptr + h_out * stride_wyn + k_offsets * stride_wyk
            wy_vals = tl.load(wy_ptrs, mask=mask_k, other=0.0)
            # dot product
            for i in range(K_BLOCK):
                acc += y_vals[i] * wy_vals[i]
        out_val = acc + tl.load(bb_ptr + h_out * stride_oh)
        out_ptr_hs = out_ptr + b * stride_ob + s * stride_os + h_out * stride_oh
        tl.store(out_ptr_hs, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only implementation. Forward launches Triton kernels and returns the final output tensor.
        No torch allocations or elementwise gating in forward. Heavy ops are performed by Triton kernels.
        Note: For correctness in the harness, the forward signature matches the original, but no torch math is used.
        """
        # We assume inputs are on CUDA; the harness provides CUDA tensors. No torch empty/contiguous in forward.

        # 1) Triple linear projection: three separate calls, each producing (B,S,H)
        # Using linear_bsh_kernel three times with slices of


def run(*args):
    return ModelNew()(*args)
