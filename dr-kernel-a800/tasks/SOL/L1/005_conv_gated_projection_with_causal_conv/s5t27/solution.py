import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # X_ptr: [M, H], W_ptr: [M_OUT, H], OUT_ptr: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    # Loop over H (input features), accumulate X * W^T
    for k0 in range(0, H, BLOCK_H):
        k = k0 + offs_n
        k_mask = k < H

        # Load X block: [BLOCK_M, BLOCK_H]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_H, BLOCK_H] where rows are output features and cols are input features
        w_ptrs = W_ptr + (k[None, :] * stride_wh + offs_n[:, None] * stride_wm)
        w = tl.load(w_ptrs, mask=k_mask[None, :] & n_mask[:, None], other=0.0)

        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, X_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_xm, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Element-wise multiply: OUT = B * X, both [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    n_mask = offs_n < H

    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bm)
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xm)

    b = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    out = b * x

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_om)
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K] (W[OUT_H, IN_H])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: [BM, BK] @ [BK, BN] -> [BM, BN]
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
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
        # x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,)
        B, S, H = x.shape
        M = B * S

        # 1) First linear projection: y_flat = F.linear(x_flat, in_proj_weight, in_proj_bias)
        # Flatten x to (M, H)
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, 3 * H), dtype=x.dtype, device=x.device)

        in_proj_H = 3 * H
        BLOCK_M = 128
        BLOCK_H = 128
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(in_proj_H, BLOCK_H))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, in_proj_H,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        )

        # 2) Split along dim=1 into B, C, x_proj
        # y_flat shape (M, 3H), split into three (M, H) tensors
        B_t = y_flat[:, :H].contiguous()          # (M, H)
        C_t = y_flat[:, H:2 * H].contiguous()     # (M, H)
        XP = y_flat[:, 2 * H:].contiguous()       # (M, H)

        # Reshape back to (B, S, H)
        B_t = B_t.view(B, S, H).contiguous()
        C_t = C_t.view(B, S, H).contiguous()
        XP = XP.view(B, S, H).contiguous()

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        BLOCK_M_mul = 128
        BLOCK_H_mul = 128
        grid_mul = (triton.cdiv(B * S, BLOCK_M_mul), triton.cdiv(H, BLOCK_H_mul))
        mul_elementwise_kernel[grid_mul](
            B_t, XP, Bx,
            B * S, H,
            B_t.stride(0), XP.stride(0), Bx.stride(0),
            BLOCK_M=BLOCK_M_mul, BLOCK_H=BLOCK_H_mul,
        )

        # 4) Grouped causal 1D convolution with PyTorch to ensure correctness
        # (Bx): (B, S, H) -> (B, H, S) with groups=H, kernel_size=4, padding=0
        Bx_perm = Bx.transpose(1, 2).contiguous()  # (B, H, S)
        conv_out = torch.nn.functional.conv1d(Bx_perm, conv_weight, conv_bias, groups=H, kernel_size=4)

        # 5) Output gating: y = C * conv_out, (B, H, S)
        y = C_t * conv_out

        # 6) Final projection: y @ out_proj_weight^T + out_proj_bias, (B, S, H)
        y_flat2 = y.transpose(1, 2).contiguous().reshape(B * S, H)  # (B*S, H)
        output_flat = torch.empty((B * S, H), dtype=x.dtype, device=x.device)
        BLOCK_M_fin = 128
        BLOCK_N_fin = 128
        BLOCK_K_fin = 128
        grid_fin = (triton.cdiv(B * S, BLOCK_M_fin), triton.cdiv(H, BLOCK_N_fin))
        final_proj_kernel[grid_fin](
            y_flat2, out_proj_weight, out_proj_bias, output_flat,
            B * S, H, H,
            y_flat2.stride(0), y_flat2.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output_flat.stride(0), output_flat.stride(1),
            BLOCK_M=BLOCK_M_fin, BLOCK_N=BLOCK_N_fin, BLOCK_K=BLOCK_K_fin,
        )
        output = output_flat.view(B, S, H).contiguous()
        return output


def run(*args):
    return ModelNew()(*args)
