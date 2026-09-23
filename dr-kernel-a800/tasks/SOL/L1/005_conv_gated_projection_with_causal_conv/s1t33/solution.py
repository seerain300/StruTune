import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y[B, S, M] = sum_h X[B, S, h] * W[M, h] + Bias[M]
# X: (B, S, H), W: (M, H), Bias: (M), Y: (B, S, M)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # reduce over H
    for h in range(0, H):
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + h * stride_x_h  # (S,1)
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)  # (S,1)

        w_ptrs = W_ptr + m_offsets[None, :] * stride_w_m + h * stride_w_h  # (M,1)
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0)  # (M,1)

        acc += x_vals * w_vals  # broadcast over M

    # add bias
    bias = tl.load(Bias_ptr + m_offsets, mask=mask_m, other=0.0)  # (M,)
    acc += bias[None, :]  # broadcast over S

    # store Y
    y_ptrs = Y_ptr + pid_b * (S * M) + s_offsets[:, None] * M + m_offsets[None, :]  # (S,M)
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# Triton kernel: Element-wise gating Bx = B * X, shapes (B, S, H)
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


# Triton kernel: Left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
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
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton kernel: Final linear projection OUT[B, S, H] = sum_h IN[B, S, h] * W[h, OUT_H] + Bias[OUT_H]
# IN: (B, S, H), W: (H, OUT_H), Bias: (OUT_H), OUT: (B, S, H)
@triton.jit
def TritonFinalLinearKernel(
    IN_ptr, W_ptr, Bias_ptr, OUT_ptr,
    B, S, H,
    stride_in_b, stride_in_s, stride_in_h,
    stride_w_in, stride_w_out,
    BLOCK_S: tl.constexpr, BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_out = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    out_offsets = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    in_offsets = tl.arange(0, BLOCK_IN)  # reduce over H, but H is small

    mask_s = s_offsets < S
    mask_out = out_offsets < H  # OUT dimension is H

    acc = tl.zeros((BLOCK_S, BLOCK_OUT), dtype=tl.float32)

    # reduce over H (IN's last dim)
    for h in range(0, H):
        in_ptrs = IN_ptr + pid_b * stride_in_b + s_offsets[:, None] * stride_in_s + h * stride_in_h  # (S,1)
        in_vals = tl.load(in_ptrs, mask=mask_s[:, None], other=0.0)  # (S,1)

        w_ptrs = W_ptr + h * stride_w_in + out_offsets[None, :] * stride_w_out  # (1, OUT)
        w_vals = tl.load(w_ptrs, mask=mask_out[None, :], other=0.0)  # (1, OUT)

        acc += in_vals * w_vals  # broadcast over OUT

    # add bias
    bias = tl.load(Bias_ptr + out_offsets, mask=mask_out, other=0.0)  # (OUT,)
    acc += bias[None, :]  # broadcast over S

    # store OUT
    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + out_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_out[None, :])


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
        # Ensure dtype and contiguity
        B, S, H = x.shape
        x = x.contiguous().to(torch.float32)

        # 1) Three linear projections using Triton
        # First projection for B
        W_B = in_proj_weight[:H, :].contiguous().to(torch.float32)  # (H, H)
        Bias_B = in_proj_bias[:H].contiguous().to(torch.float32)    # (H,)
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonLinearKernel[(B, triton.cdiv(S, 32), triton.cdiv(H, 32))](  # grid
            x, W_B, Bias_B, B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W_B.stride(0), W_B.stride(1),
            BLOCK_S=32, BLOCK_M=32, BLOCK_H=32, num_warps=4, num_stages=2
        )

        # Second projection for C
        W_C = in_proj_weight[H : 2 * H, :].contiguous().to(torch.float32)  # (H, H)
        Bias_C = in_proj_bias[H : 2 * H].contiguous().to(torch.float32)     # (H,)
        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonLinearKernel[(B, triton.cdiv(S, 32), triton.cdiv(H, 32))](  # grid
            x, W_C, Bias_C, C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W_C.stride(0), W_C.stride(1),
            BLOCK_S=32, BLOCK_M=32, BLOCK_H=32, num_warps=4, num_stages=2
        )

        # Third projection for x_proj
        W_X = in_proj_weight[2 * H :, :].contiguous().to(torch.float32)     # (H, H)
        Bias_X = in_proj_bias[2 * H :].contiguous().to(torch.float32)       # (H,)
        x_proj_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonLinearKernel[(B, triton.cdiv(S, 32), triton.cdiv(H, 32))](  # grid
            x, W_X, Bias_X, x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            W_X.stride(0), W_X.stride(1),
            BLOCK_S=32, BLOCK_M=32, BLOCK_H=32, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonGateKernel[(B, triton.cdiv(S, 32), triton.cdiv(H, 32))](  # grid
            B_out, x_proj_out, Bx,
            B, S, H,
            BLOCK_S=32, BLOCK_H=32, num_warps=4, num_stages=2
        )

        # 3) Left-pad along sequence by 3 for causal conv
        Bx_pad = torch.empty((B, H, S + 3), device=x.device, dtype=torch.float32)
        TritonPadLeftKernel[(B, H, triton.cdiv(S + 3, 128))](
            Bx,
            Bx_pad,
            B, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: use PyTorch F.conv1d for robustness
        # conv_weight: (H, 1, 4), conv_bias: (H), groups=H
        conv_out = torch.nn.functional.conv1d(
            Bx_pad, conv_weight, conv_bias, stride=1, padding=3, groups=H
        )  # shape: (B, H, S)

        # 5) Output gating: y = C * conv_out (elementwise). For robustness, use PyTorch multiply.
        # Shapes: C (B, S, H), conv_out (B, H, S)
        # We need broadcasting: C[:, :, :] * conv_out[:, :, :]
        # Create y as zeros and fill elementwise.
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        # Implement elementwise multiply via broadcasting in PyTorch (simple and correct):
        # Note: PyTorch requires broadcasting; we do it here for correctness.
        y = C_out.transpose(1, 2) * conv_out  # (B, H, S)

        # 6) Final output projection: OUT[B, S, H] = y @ out_proj_weight.T + out_proj_bias
        # y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H)
        # We need OUT = y @ W.T + bias
        # Implement TritonFinalLinearKernel:
        # IN is y transposed to (B, S, H) for kernel: IN_ptr will point to y.contiguous() as (B, H, S) then reindex.
        # However TritonFinalLinearKernel expects IN as (B, S, H). We can create IN by transposing y to (B, S, H):
        # Let's make IN explicit:
        IN_final = y.transpose(1, 2).contiguous()  # (B, S, H)
        OUT_final = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # W is (H, H), IN_FINAL is (B, S, H), bias (H)
        TritonFinalLinearKernel[(B, triton.cdiv(S, 32), triton.cdiv(H, 32))](  # grid
            IN_final, out_proj_weight, out_proj_bias, OUT_final,
            B, S, H,
            IN_final.stride(0), IN_final.stride(1), IN_final.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            BLOCK_S=32, BLOCK_IN=32, BLOCK_OUT=32, num_warps=4, num_stages=2
        )

        return OUT_final


def run(*args):
    return ModelNew()(*args)
