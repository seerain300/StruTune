import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y[B, S, M] = X[B, S, H] @ W[M, H]^T + bias[M]
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    Bsz, S, H, M,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = tl.arange(0, BLOCK_H)

    mask_m = m_offsets < M
    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduce over H
    for i in range(0, BLOCK_H):
        hi = h_offsets[i]
        x_ptrs = X_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + hi * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # (BLOCK_S, 1)

        w_ptrs = W_ptr + m_offsets * stride_w_m + hi * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)

        acc += x_vals * w_vals[None, :]

    # Add bias
    bias_ptrs = Bias_ptr + m_offsets
    bias_vals = tl.load(bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)
    acc += bias_vals[None, :]

    out_ptrs = Out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m_offsets[None, :] * stride_out_h
    store_mask = (mask_s[:, None]) & (mask_m[None, :])
    tl.store(out_ptrs, acc, mask=store_mask)


# Triton kernel: Element-wise gating Bx = B * x_proj over (B, S, H)
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Grouped causal 1D convolution (groups=H, kernel_size=4)
# Inputs: Bx of shape (B, H, S). conv_weight: (H, 1, 4), conv_bias: (H). groups=H.
# Output: conv_out of shape (B, H, S).
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,  # conv_weight strides
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    s_start = pid_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Kernel size = 4 (k in 0..3), causal padding handled via masked loads
    for k in range(0, 4):
        idx = s_offsets + k
        load_mask = (idx >= 0) & (idx < S) & mask_s
        bx_ptrs = Bx_ptr + b * stride_bx_b + h * stride_bx_h + idx * stride_bx_s
        bx_vals = tl.load(bx_ptrs, mask=load_mask, other=0.0).to(tl.float32)

        # conv_weight[h, 0, k] accessed linearly: conv_weight_ptr[h * 4 + k]
        w_ptr = conv_weight_ptr + h * 4 + k
        w_val = tl.load(w_ptr).to(tl.float32)
        acc += bx_vals * w_val

    # Add bias
    bias_ptr = conv_bias_ptr + h
    b_val = tl.load(bias_ptr).to(tl.float32)
    acc += b_val

    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + s_offsets * stride_out_s
    tl.store(out_ptrs, acc, mask=mask_s)


# Triton kernel: Final linear projection Y[B, S, M] = X[B, S, H] @ W[M, H]^T + bias[M]
@triton.jit
def TritonFinalLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    Bsz, S, H, M,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = tl.arange(0, BLOCK_H)

    mask_m = m_offsets < M
    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    for i in range(0, BLOCK_H):
        hi = h_offsets[i]
        x_ptrs = X_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + hi * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # (BLOCK_S, 1)

        w_ptrs = W_ptr + m_offsets * stride_w_m + hi * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)

        acc += x_vals * w_vals[None, :]

    bias_ptrs = Bias_ptr + m_offsets
    bias_vals = tl.load(bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)
    acc += bias_vals[None, :]

    out_ptrs = Out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m_offsets[None, :] * stride_out_h
    store_mask = (mask_s[:, None]) & (mask_m[None, :])
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # x: (B, S, H)
        B, S, H = x.shape
        assert H == self.hidden_size

        # 1) Three linear projections via TritonLinearKernel
        B_out = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        C_out = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        x_proj_out = torch.empty((B, S, H), dtype=torch.float32, device=x.device)

        BLOCK_S = 128
        BLOCK_H = 128
        grid_linear = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))

        TritonLinearKernel[grid_linear](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S, BLOCK_H
        )

        TritonLinearKernel[grid_linear](
            x, in_proj_weight[H:2 * H, :], in_proj_bias[H:2 * H], C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S, BLOCK_H
        )

        TritonLinearKernel[grid_linear](
            x, in_proj_weight[2 * H:3 * H, :], in_proj_bias[2 * H:3 * H], x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_S, BLOCK_H
        )

        # 2) Element-wise gating Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_gate = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
        TritonGateKernel[grid_gate](
            B_out, x_proj_out, Bx,
            B, S, H,
            128, 128
        )

        # 3) Grouped causal conv with kernel_size=4, groups=H
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
        # conv_weight is (H, 1, 4): contiguous => stride_w_h = 4, stride_w_k = 1
        BLOCK_S_conv = 128
        grid_conv = (B, H, triton.cdiv(S, BLOCK_S_conv))
        TritonGroupedCausalConvKernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), 1,  # stride_w_k = 1, stride_w_h = H*1*4 = 4
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S_conv
        )

        # 4) Output gating: y = C * conv_out (elementwise). This is simple, and Triton can do it too.
        # Implement gating in Triton to keep all compute in Triton.
        y = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_gate2 = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
        TritonGateKernel[grid_gate2](
            C_out, conv_out.transpose(-1, -2).contiguous(), y,
            B, S, H,
            128, 128
        )
        # Note: We need y = C * conv_out where C is (B, S, H) and conv_out is (B, H, S).
        # TritonGateKernel expects


def run(*args):
    return ModelNew()(*args)
