import torch
import triton
import triton.language as tl

# Triton kernel: in_proj
# Computes BCx[b, s, m] = sum_h x[b, s, h] * in_proj_weight[m, h] + in_proj_bias[m]
# Inputs:
#   x_ptr: (B, S, H)
#   weight_ptr: (M, H), M = 3*H
#   bias_ptr: (M,)
# Output:
#   out_ptr: (B, S, M), float32
@triton.jit
def in_proj_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    grid = (B, S, triton.cdiv(M, BLOCK_M))
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # accumulator for this (b, s, m_block)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over H dimension (compile-time constant loop)
    for h in range(0, H):
        # load x[b, s, h] (scalar)
        x_val = tl.load(x_ptr + b * (S * H) + s * H + h)
        # load weight[m, h] vector across m_offsets
        w_ptrs = weight_ptr + m_offsets * H + h
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0)
        acc += x_val * w_vals

    # add bias
    b_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # store to out[b, s, m]
    out_ptrs = out_ptr + b * (S * M) + s * M + m_offsets
    tl.store(out_ptrs, acc, mask=mask_m)


# Triton kernel: elementwise gate multiply
# Computes Out[b, s, h] = A[b, s, h] * B[b, s, h]
@triton.jit
def gate_mul_kernel(
    A_ptr, B_ptr, Out_ptr,
    Bsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    grid = (Bsz, S, triton.cdiv(H, BLOCK_H))
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    A_row = A_ptr + b * (S * H) + s * H
    B_row = B_ptr + b * (S * H) + s * H
    Out_row = Out_ptr + b * (S * H) + s * H

    A_vals = tl.load(A_row + h_offsets, mask=mask_h, other=0.0)
    B_vals = tl.load(B_row + h_offsets, mask=mask_h, other=0.0)
    Out_vals = A_vals * B_vals
    tl.store(Out_row + h_offsets, Out_vals, mask=mask_h)


# Triton kernel: grouped causal 1D convolution (groups=H, kernel_size=4)
# Input: X_padded: (B, S_padded, H), S_padded = S + K - 1
# Weight: (H, 1, 4), bias: (H,)
# Output: out_conv: (B, H, S) float32
@triton.jit
def conv1d_groupsH_kernel(
    X_ptr,        # *float32, (B, S_padded, H)
    weight_ptr,   # *float32, (H, 4), we pass weight[c, k] for c and k fixed
    bias_ptr,     # *float32, (H,)
    out_ptr,      # *float32, (B, H, S)
    B: tl.constexpr, H: tl.constexpr, S_padded: tl.constexpr, K: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    # grid over (B, H, ceil(S / BLOCK_T))
    b = tl.program_id(0)
    c = tl.program_id(1)
    t_block = tl.program_id(2)

    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over kernel taps k in [0, K)
    for k in range(0, K):
        t_in = t_offsets + k
        mask_t_in = (t_in < S_padded) & mask_t
        # load X[b, t_in, c]
        x_ptrs = X_ptr + b * (S_padded * H) + t_in * H + c
        x_vals = tl.load(x_ptrs, mask=mask_t_in, other=0.0)
        # weight per channel c for this k
        w_val = tl.load(weight_ptr + c * 4 + k, mask=True, other=0.0)
        acc += x_vals * w_val

    # add bias
    b_val = tl.load(bias_ptr + c, mask=True, other=0.0)
    acc += b_val

    # store out[b, c, t]
    out_ptrs = out_ptr + b * (H * S) + c * S + t_offsets
    tl.store(out_ptrs, acc, mask=mask_t)


# Triton kernel: out_proj (matmul-like) y -> output
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def out_proj_kernel(
    y_ptr, out_w_ptr, out_b_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    grid = (B, S, triton.cdiv(H, BLOCK_H))
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)

    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # loop over h2
    for h2 in range(0, H):
        y_val = tl.load(y_ptr + b * (S * H) + s * H + h2)
        w_vals = tl.load(out_w_ptr + h2 * H + h_offsets, mask=mask_h, other=0.0)
        acc += y_val * w_vals

    bias_vals = tl.load(out_b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals

    out_row = out_ptr + b * (S * H) + s * H
    tl.store(out_row + h_offsets, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H), float32
        in_proj_weight: (3*H, H), float32
        in_proj_bias: (3*H,), float32
        conv_weight: (H, 1, 4) -> we use (H, 4) by reshaping, float32
        conv_bias: (H,), float32
        out_proj_weight: (H, H), float32
        out_proj_bias: (H,), float32
        Returns: (B, S, H), float32
        """
        device = x.device
        B, S, H = x.shape
        M = 3 * H  # in_proj outputs 3*H channels

        # 1) Triton in_proj: compute BCx of shape (B, S, M)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)

        BLOCK_M = 128
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x.contiguous(), in_proj_weight.contiguous(), in_proj_bias.contiguous(),
            BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split into B, C, x_proj along last dim of size H
        B_t = BCx[:, :, :H]       # (B, S, H)
        C_t = BCx[:, :, H:2*H]    # (B, S, H)
        x_proj = BCx[:, :, 2*H:]  # (B, S, H)

        # 3) Triton gating: Bx = B_t * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_gate = (B, S, triton.cdiv(H, 64))
        gate_mul_kernel[grid_gate](
            B_t.contiguous(), x_proj.contiguous(), Bx,
            B=B, S=S, H=H, BLOCK_H=64,
            num_warps=2, num_stages=2
        )

        # 4) Left-pad for causal conv along sequence by K-1 = 3
        # Input for conv is (B, S+3, H)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, S+3, H)

        # 5) Triton grouped causal conv with groups=H, kernel_size=4
        S_padded = S + 3
        K = 4
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        # weight reshaped to (H, 4) for easy indexing
        conv_weight_4 = conv_weight.view(H, -1).contiguous()  # (H, 4)

        BLOCK_T = 128
        grid_conv = (B, H, triton.cdiv(S, BLOCK_T))
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_4, conv_bias.contiguous(), conv_out,
            B=B, H=H, S_padded=S_padded, K=K, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out
        y = C_t * conv_out  # (B, H, S)

        # 7) Transpose back to (B, S, H) for final projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H), float32

        # 8) Triton out_proj: y_T -> output
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_H = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_out](
            y_T.contiguous(), out_proj_weight.contiguous(), out_proj_bias.contiguous(),
            output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
