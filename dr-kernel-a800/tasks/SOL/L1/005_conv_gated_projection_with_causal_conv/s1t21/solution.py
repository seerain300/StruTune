import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y[b, s, m] = sum_h X[b, s, h] * W[m, h] + bias[m]
# Shapes:
#   x: [B, S, H], contiguous
#   w: [M, H], contiguous (M = H for each of the three groups)
#   bias: [M], contiguous
#   out: [B, S, M], contiguous (we set M=H)
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m  # m is one of {0..H-1}

    # Accumulator for this (b, s) row over M
    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    # Loop over H (reduction dimension)
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load X[b, s, h] as [BLOCK_S, BLOCK_H]
        x_ptrs = x_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + h_offsets[None, :] * stride_x_h
        x_tile = tl.load(x_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)

        # Load W[m, h] as [BLOCK_H], then broadcast to [BLOCK_S, BLOCK_H]
        w_ptrs = w_ptr + m * stride_w_m + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
        w_tile = w_tile[None, :]  # broadcast along s dimension

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Add bias[m]
    bias_val = tl.load(bias_ptr + m)
    acc += bias_val  # broadcast over s

    # Store to out[b, s, m]
    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=s_offsets[:, None] < S)


# Triton kernel: Element-wise gating: OUT = B * X, both are (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Left-pad along sequence by PAD (here PAD=3) -> OUT[B, H, S + PAD]
# Inputs: Bx [B, H, S] contiguous; OUT [B, H, S+PAD] contiguous
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # Write zeros for the first PAD columns
    for i in range(0, PAD):
        tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s, 0.0)

    # Copy Bx[b, h, :] into OUT[b, h, PAD:] starting at s_out_start
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton kernel: Grouped causal 1D convolution (groups=H), kernel_size=4
# Input: Bx_pad [B, H, S+PAD]; conv_weight [H, 1, 4]; conv_bias [H]
# Output: conv_out [B, H, S]
@triton.jit
def TritonCausalConv1dGroupsKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD, K,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_b
    h = pid_h

    # Iterate over sequence positions
    for s_out in range(0, S):
        acc = 0.0
        # Unrolled loop over kernel taps (K=4)
        for k in range(0, K):
            s_in = s_out + k
            # If s_in >= S+PAD, padded zeros handled by conv
            val = tl.load(Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            w = tl.load(conv_weight_ptr + h * stride_w_h + k * stride_w_k)
            acc += val * w
        bias = tl.load(conv_bias_ptr + h)
        acc += bias
        tl.store(out_ptr + b * stride_out_b + h * stride_out_h + s_out * stride_out_s, acc)


# Triton kernel: Final projection OUT[b, s, m] = sum_h Y[b, s, h] * W[m, h] + bias[m]
# Shapes:
#   Y: [B, S, H], contiguous
#   W: [M, H], contiguous (M=H)
#   bias: [M]
#   OUT: [B, S, M], contiguous
@triton.jit
def TritonLinearFinalKernel(
    Y_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_y_b, stride_y_s, stride_y_m,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m

    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        y_ptrs = Y_ptr + b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_m
        y_tile = tl.load(y_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_H]

        w_ptrs = w_ptr + m * stride_w_m + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
        w_tile = w_tile[None, :]  # broadcast across s

        acc += tl.dot(y_tile, w_tile)

    bias_val = tl.load(bias_ptr + m)
    acc += bias_val  # broadcast over s

    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=s_offsets[:, None] < S)


# ModelNew: forward uses only Triton kernels (no torch ops for compute)
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
        # Shapes
        B, S, H = x.shape
        assert in_proj_weight.shape == (3 * H, H), "in_proj_weight must have shape (3*H, H)"
        assert in_proj_bias.shape == (3 * H,), "in_proj_bias must have shape (3*H,)"
        assert conv_weight.shape[0] == H and conv_weight.shape[1] == 1 and conv_weight.shape[2] == 4, "conv_weight must be (H, 1, 4)"
        assert conv_bias.shape == (H,), "conv_bias must have shape (H,)"
        assert out_proj_weight.shape == (H, H), "out_proj_weight must have shape (H, H)"
        assert out_proj_bias.shape == (H,), "out_proj_bias must have shape (H,)"

        # Ensure contiguity
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        # Device: ensure Triton runs on same device
        device = x.device

        # 1) Three linear projections (Triton)
        # Output tensors (B, S, H)
        B_out = torch.empty((B, S, H), dtype=torch.float32, device=device)
        C_out = torch.empty((B, S, H), dtype=torch.float32, device=device)
        x_proj_out = torch.empty((B, S, H), dtype=torch.float32, device=device)

        # Kernel launch config
        BLOCK_S = 128
        BLOCK_H = 32

        # First projection: B
        TritonLinearProjectionKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, in_proj_weight[:H, :], in_proj_bias[:H],
            B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Second projection: C
        TritonLinearProjectionKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H],
            C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Third projection: x_proj
        TritonLinearProjectionKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H],
            x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        TritonGateKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            B_out, x_proj_out, Bx,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 3) Left-pad along sequence by PAD=3
        PAD = 3
        Bx_pad = torch.empty((B, H, S + PAD), dtype=torch.float32, device=device)
        TritonPadLeftKernel[(B, H, triton.cdiv(S + PAD, 128))](
            Bx,
            Bx_pad,
            B, H, S, PAD,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution (groups=H, kernel_size=4) via Triton
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        TritonCausalConv1dGroupsKernel[(B, H)](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, S, PAD, 4,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1
        )

        # 5) Output gating: y = C * conv_out
        # Shapes: C: (B, S, H), conv_out: (B, H, S)
        # y has shape (B, S, H)
        y = torch.empty((B, S, H), dtype=torch.float32, device=device)
        TritonGateKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            C_out, conv_out, y,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 6) Final projection: F.linear(y, out_proj_weight, out_proj_bias)
        # Implement with TritonLinearFinalKernel: OUT[b, s, m] = sum_h y[b, s, h] * out_proj_weight[m, h] + bias[m]
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        TritonLinearFinalKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
