import torch
import triton
import triton.language as tl


# Kernel 1: In-projection linear (y = x @ W^T + bias)
# x: [B, S, H] flattened to [M, H], M = B * S
# W: [M_out, H]
# bias: [M_out]
# Output: [M, M_out]
@triton.jit
def in_proj_linear_kernel(
    x_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, H, M_out,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]

        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wm + k_ids[:, None] * stride_wh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, tl.trans(w))

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# Triton element-wise multiply: out = a * b
@triton.jit
def mul_elementwise_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * b, mask=mask)


# Kernel 2: Grouped causal conv1d with groups=H, kernel_size=4 (PAD=3), pre-padding
# Input Bx_padded: [B, H, L_padded] where L_padded = L + PAD. We'll pass flattened as [N, H, L_padded] where N=B*S.
# conv_weight: [H, K=4], conv_bias: [H]
# Output: [N, H] (we'll reshape to [B, S, H] on host)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, W_ptr, BIAS_ptr, Out_ptr,
    N, H, L_padded, PAD, K,
    stride_bxn, stride_bxh, stride_bxl,
    stride_wh, stride_wk,
    stride_on, stride_oh,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Grid: (N, H)
    n = tl.program_id(0)  # flattened index over B*S
    h = tl.program_id(1)  # channel index

    offs_t = tl.arange(0, BLOCK_T)
    t = n * H + h  # combined index to iterate output positions
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = (t + offs_t) < L_padded  # valid positions in padded sequence

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Reduction over kernel taps
    for k in range(0, K):
        t_in = t - k  # because Bx_padded has zeros on left, no need for +PAD here
        valid = (t_in >= 0) & (t_in < L_padded)
        # For padded flattened indexing: idx = n*(H*L_padded) + h*L_padded + t_in
        idx = n * (H * L_padded) + h * L_padded + t_in
        val = tl.load(Bx_ptr + idx, mask=valid, other=0.0)
        w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)  # scalar weight
        acc += val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + h)
    acc += bias

    # Store to Out[n, h] (vectorized across offs_t)
    out_idx = n * H + h
    tl.store(Out_ptr + out_idx + offs_t * stride_oh, acc, mask=t_mask)


# Kernel 3: Out-projection linear (final)
# y: [B, S, H] flattened to [M, H], W_out: [H, H], bias_out: [H]
# Output: [M, H]
@triton.jit
def out_proj_linear_kernel(
    y_ptr, W_out_ptr, BIAS_out_ptr, Out_ptr,
    M, H,
    stride_ym, stride_yh,
    stride_wom, stride_woh,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        y = tl.load(
            y_ptr + offs_m[:, None] * stride_ym + k_ids[None, :] * stride_yh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        w = tl.load(
            W_out_ptr + offs_n[None, :] * stride_wom + k_ids[:, None] * stride_woh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(y, tl.trans(w))

    bias = tl.load(BIAS_out_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (M_out, H), M_out=3*H
        in_proj_bias: (M_out,)
        conv_weight: (H, K=4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        device = x.device
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]

        # 1) In-projection: y_flat = x_flat @ W^T + bias
        x_flat = x.reshape(B * S, H).contiguous()
        y_flat = torch.empty((B * S, M_out), dtype=x.dtype, device=device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid_in = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            B * S, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )
        y = y_flat.view(B, S, M_out).contiguous()

        # 2) Split y into B, C, x_proj along last dim
        # Note: y has shape (B, S, M_out), chunk along dim=2 into 3 parts of size H
        # PyTorch version uses y[:, :, :H], y[:, :, H:2H], y[:, :, 2H:3H]
        B_part = y[:, :, :H].contiguous()           # (B, S, H)
        C_part = y[:, :, H:2 * H].contiguous()      # (B, S, H)
        x_proj = y[:, :, 2 * H:].contiguous()       # (B, S, H)

        # 3) Element-wise gating: Bx = B_part * x_proj
        Bx = B_part * x_proj

        # 4) Grouped causal conv with pre-padding (PAD=3), kernel_size=4, groups=H
        # Pre-pad along S by 3 zeros to match conv1d with (3, 0) padding
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, S+3, H), pad only left
        # Flatten to [N, H, L_padded] where N=B*S
        L_padded = S + 3
        Bx_flat = Bx_padded.reshape(B * S, H, L_padded).contiguous()
        conv_out_flat = torch.empty((B * S, H), dtype=x.dtype, device=device)

        grid_conv = (B * S, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_flat, conv_weight, conv_bias, conv_out_flat,
            B * S, H, L_padded, PAD=3, K=4,
            stride_bxn=Bx_flat.stride(0), stride_bxh=Bx_flat.stride(1), stride_bxl=Bx_flat.stride(2),
            stride_wh=conv_weight.stride(0), stride_wk=conv_weight.stride(1),
            stride_on=Bx_flat.stride(0), stride_oh=H,
            BLOCK_C=1, BLOCK_T=256,
            num_warps=4, num_stages=2
        )
        conv_out = conv_out_flat.view(B, S, H).contiguous()

        # 5) Output gating: y = C_part * conv_out
        y_gate = C_part * conv_out

        # 6) Final projection: y_gate @ out_proj_weight^T + out_proj_bias
        M = B * S
        y_gate_flat = y_gate.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), dtype=x.dtype, device=device)
        BLOCK_M_out, BLOCK_N_out, BLOCK_K_out = 64, 64, 32
        grid_out = (triton.cdiv(M, BLOCK_M_out), triton.cdiv(H, BLOCK_N_out))
        out_proj_linear_kernel[grid_out](
            y_gate_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H,
            y_gate_flat.stride(0), y_gate_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_M_out, BLOCK_N=BLOCK_N_out, BLOCK_K=BLOCK_K_out,
            num_warps=4, num_stages=2
        )

        output = out_flat.view(B, S, H).contiguous()
        return output


def run(*args):
    return ModelNew()(*args)
