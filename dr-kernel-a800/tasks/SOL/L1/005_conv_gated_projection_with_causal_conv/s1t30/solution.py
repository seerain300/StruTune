import torch
import triton
import triton.language as tl


# Triton Linear Projection Kernel: out[B, S, M] = x @ W[M, :].T + bias[M]
# - x: (B, S, H), contiguous (we'll pass as pointer with strides, but we use contiguous in forward)
# - W: (M, H), contiguous, M in {H, H, H} for the three projections
# - bias: (M)
# - out: (B, S, M), contiguous
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    # program ids: tile over (b, s_block, h_block)
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # masks
    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # accumulator for each (b, s, h) output
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # reduce over K dimension (H)
    for k_start in range(0, H, BLOCK_H):
        kh_offsets = k_start + tl.arange(0, BLOCK_H)
        mask_k = kh_offsets < H

        # load x tile: shape (BLOCK_S, BLOCK_H)
        x_ptrs = x_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + kh_offsets[None, :] * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)
        x_vals = x_vals.to(tl.float32)

        # load w tile: shape (BLOCK_H, BLOCK_H)
        w_ptrs = w_ptr + h_offsets[:, None] * stride_w_m + kh_offsets[None, :] * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)
        w_vals = w_vals.to(tl.float32)

        # accumulate: x (BLOCK_S, BLOCK_H) @ w^T (BLOCK_H, BLOCK_H) -> (BLOCK_S, BLOCK_H)
        # We do outer-product accumulation: for each kk in BLOCK_H, add x_vals[:, kk][:, None] * w_vals[:, kk][None, :]
        for kk in range(BLOCK_H):
            if tl.any(mask_k[kk]):  # guard if block reaches end
                x_col = x_vals[:, kk]  # (BLOCK_S,)
                w_col = w_vals[:, kk]  # (BLOCK_H,)
                acc += x_col[:, None] * w_col[None, :]

    # add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # store result
    out_ptrs = out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton Element-wise Gating: OUT = B * X
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton Pad Left Kernel: out_pad[b, h, s_out] where s_out in [0, S+PAD), with zeros at s_out < PAD
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
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton Grouped Causal Convolution (groups=H, kernel_size=4):
# Input: Bx_pad [B, H, S+PAD], weight [H, 1, 4], bias [H]
# Output: conv_out [B, H, S]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # accumulator for outputs
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # compute conv for each output s in the block
    for i in range(0, BLOCK_S):
        s_out = s_offsets[i]
        if s_out < S:
            # sum over k=0..3: conv is causal, so valid indices are s_out + k in [0, S_out)
            for k in range(4):
                in_s = s_out + k
                if in_s < S_out:
                    val = tl.load(Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + in_s * stride_bx_s)
                    w = tl.load(conv_weight_ptr + h * stride_w_h + k * stride_w_k)
                    acc[i] += val * w

    # add bias
    bias = tl.load(conv_bias_ptr + h)
    acc += bias

    # store
    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + s_offsets * stride_out_s
    tl.store(out_ptrs, acc, mask=mask_s)


# Triton Element-wise Multiply: OUT = C * IN, shapes (B, S, H)
@triton.jit
def TritonMulKernel(
    C_ptr, IN_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    c_ptrs = C_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    in_ptrs = IN_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    c_vals = tl.load(c_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    in_vals = tl.load(in_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = c_vals * in_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Implement the original fused computation with Triton kernels:
        1) Triple linear projection: x -> (B, S, H) via in_proj_weight[:H,:], [:H, :], [H:2H,:]
        2) Element-wise gating: Bx = B * x_proj
        3) Grouped causal 1D convolution (groups=H, kernel_size=4) on Bx with padding 3 on left
        4) Output gating: y = C * conv_out
        5) Final output projection: linear(y, out_proj_weight, out_proj_bias)
        """
        # Ensure inputs are on CUDA and contiguous
        assert x.is_cuda, "Input x must be on CUDA device"
        B, S, H = x.shape

        # 1) Three linear projections using Triton
        # Prepare outputs for each projection
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        x_proj_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # Tile sizes
        BLOCK_S = 64
        BLOCK_H = 64

        # First projection: B = F.linear(x, in_proj_weight[:H,:], in_proj_bias[:H])
        W1 = in_proj_weight[:H, :].contiguous()
        b1 = in_proj_bias[:H].contiguous()
        TritonLinearProjectionKernel[(B, _ceil_div(S, BLOCK_S), _ceil_div(H, BLOCK_H))](
            x, W1, b1, B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # Second projection: C = F.linear(x, in_proj_weight[H:2H,:], in_proj_bias[H:2H])
        W2 = in_proj_weight[H:2 * H, :].contiguous()
        b2 = in_proj_bias[H:2 * H].contiguous()
        TritonLinearProjectionKernel[(B, _ceil_div(S, BLOCK_S), _ceil_div(H, BLOCK_H))](
            x, W2, b2, C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # Third projection: x_proj = F.linear(x, in_proj_weight[2H:3H,:], in_proj_bias[2H:3H])
        W3 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b3 = in_proj_bias[2 * H:3 * H].contiguous()
        TritonLinearProjectionKernel[(B, _ceil_div(S, BLOCK_S), _ceil_div(H, BLOCK_H))](
            x, W3, b3, x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W3.stride(0), W3.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonGateKernel[(B, _ceil_div(S, BLOCK_S), _ceil_div(H, BLOCK_H))](
            B_out, x_proj_out, Bx,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 3) Left-pad along S by PAD=3 for causal conv: Bx_pad [B, H, S+3]
        Bx_pad = torch.empty((B, H, S + 3), device=x.device, dtype=torch.float32)
        TritonPadLeftKernel[(B, H, _ceil_div(S + 3, BLOCK_S))](
            Bx, Bx_pad,
            B, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=BLOCK_S
        )

        # 4) Grouped causal 1D convolution with groups=H, kernel_size=4
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        # Ensure conv_weight is contiguous and shape (H, 1, 4)
        conv_weight_c = conv_weight.contiguous()
        # Weight strides: conv_weight_c has shape (H, 1, 4) => strides (4, 4, 1)
        stride_w_h = conv_weight_c.stride(0)
        stride_w_k = conv_weight_c.stride(2)

        TritonGroupedCausalConvKernel[(B, H, _ceil_div(S, BLOCK_S))](
            Bx_pad, conv_weight_c, conv_bias.contiguous(), conv_out,
            B, H, S, 3,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            stride_w_h, stride_w_k,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        TritonMulKernel[(B, _ceil_div(S, BLOCK_S), _ceil_div(H, BLOCK_H))](
            C_out, conv_out, y,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 6) Final output projection using PyTorch for correctness (Triton implementation possible, but cuBLAS is fast)
        # y shape is (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H)
        # We need output of shape (B, S, H). F.linear handles (B, S, H) input if we transpose y back to (B, S, H).
        # However, our y is (B, H, S). To match original, we need to apply linear on each (b, s, :) slice.
        # A safe way is to use F.linear on (B, S, H) by transposing y to (B, S, H). But y has shape (B, H, S).
        # Instead, we can implement as F.linear(y.transpose(-1, -2).contiguous(), out_proj_weight, out_proj_bias)
        # However, original y is (B, H, S), and F.linear expects (B, S, H) for linear with out_proj_weight (H, H).
        # To avoid confusion, we can directly do: y_T = y.transpose(1, 2).contiguous() -> (B, S, H)
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)

        output = torch.nn.functional.linear(y_T, out_proj_weight, out_proj_bias)
        return output


def run(*args):
    return ModelNew()(*args)
