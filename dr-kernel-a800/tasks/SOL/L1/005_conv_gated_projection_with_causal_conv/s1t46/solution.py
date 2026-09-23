import torch
import triton
import triton.language as tl


# Triton kernel: linear-like projection out[B, S, M] = x @ W[M, :].T + bias[M]
# x: (B, S, H) contiguous; W: (M, H) contiguous; bias: (M); out: (B, S, M)
@triton.jit
def TritonLinearKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduce over feature dimension H
    for h in range(0, H):
        # Load X[b, s, h] for current b and s block
        x_ptrs = x_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + h * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # (BLOCK_S, 1)

        # Load W[m, h] for current m block
        w_ptrs = w_ptr + m_offsets[None, :] * stride_w_m + h * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)  # (1, BLOCK_M)

        # Accumulate outer product
        acc += x_vals * w_vals  # (BLOCK_S, BLOCK_M)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)
    acc += bias_vals[None, :]

    # Store to out[b, s, m]
    out_ptrs = out_ptr + pid_b * stride_o_b + s_offsets[:, None] * stride_o_s + m_offsets[None, :] * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: final linear out[B, S, H] = y @ out_proj_weight.T + out_proj_bias
@triton.jit
def TritonFinalLinearKernel(
    Y_ptr, out_w_ptr, out_b_ptr, OUT_ptr,
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_m = m_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Reduce over H
    for h in range(0, H):
        y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + h * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # (BLOCK_S, 1)

        w_ptrs = out_w_ptr + m_offsets[None, :] * stride_w_m + h * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)  # (1, BLOCK_H)

        acc += y_vals * w_vals

    # Add bias
    bias_vals = tl.load(out_b_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + m_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H)
        in_proj_bias: (3H)
        conv_weight: (H, 1, 4)
        conv_bias: (H)
        out_proj_weight: (H, H)
        out_proj_bias: (H)
        """
        assert x.shape[2] == self.hidden_size, "hidden_size mismatch"
        B, S, H = x.shape

        # 1) Three linear projections using TritonLinearKernel
        grid = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))

        # B = x @ in_proj_weight[:H, :].T + in_proj_bias[:H]
        B_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearKernel[grid](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            128, 128,
        )

        # C = x @ in_proj_weight[H:2H, :].T + in_proj_bias[H:2H]
        C_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearKernel[grid](
            x, in_proj_weight[H:2 * H, :], in_proj_bias[H:2 * H], C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            128, 128,
        )

        # x_proj = x @ in_proj_weight[2H:3H, :].T + in_proj_bias[2H:3H]
        x_proj_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearKernel[grid](
            x, in_proj_weight[2 * H:3 * H, :], in_proj_bias[2 * H:3 * H], x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            128, 128,
        )

        # 2) Element-wise gating: Bx = B * x_proj via TritonGateKernel
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid_gate = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
        TritonGateKernel[grid_gate](
            B_out, x_proj_out, Bx,
            B, S, H,
            128, 128,
        )

        # 3) Grouped causal 1D convolution (PyTorch for robust correctness)
        # Bx: (B, S, H), conv_weight: (H, 1, 4), conv_bias: (H), groups=H
        # Pad left by 3 for kernel_size=4 (causal)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, S, H) -> pad last dim (sequence)
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, stride=1, groups=H
        )  # (B, H, S)

        # 4) Output gating: y = C * conv_out (elementwise)
        # C: (B, S, H), conv_out: (B, H, S) -> elementwise multiply yields (B, S, H)
        y = C_out * conv_out.transpose(-1, -2)

        # 5) Final output projection using TritonFinalLinearKernel
        out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid_final = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
        TritonFinalLinearKernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            128, 128,
        )
        return out


def run(*args):
    return ModelNew()(*args)
