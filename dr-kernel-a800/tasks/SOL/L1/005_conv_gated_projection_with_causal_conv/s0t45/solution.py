import torch
import triton
import triton.language as tl


# 1) in_proj_kernel: computes BCx[b, s, m] = sum_h x[b, s, h] * in_proj_weight[m, h] + in_proj_bias[m]
@triton.jit
def in_proj_kernel(
    x_ptr,            # *float32, (B, S, H)
    in_w_ptr,         # *float32, (3H, H)
    in_b_ptr,         # *float32, (3H,)
    out_ptr,          # *float32, (B, S, 3H)
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    H: tl.constexpr,  # int
    M: tl.constexpr,  # 3*H
    BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)
    m_start = m_block * BLOCK_M
    m = m_start + tl.arange(0, BLOCK_M)
    mask_m = m < M

    # Accumulator for this (b, s) row across BLOCK_M outputs
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over h dimension (hidden size)
    for h in range(0, H):
        # x[b, s, h] index: ((b * S + s) * H + h)
        x_val = tl.load(x_ptr + (b * S + s) * H + h, mask=mask_m, other=0.0)
        # in_w[m, h] index: m * H + h
        w_val = tl.load(in_w_ptr + m * H + h, mask=mask_m, other=0.0)
        acc += x_val * w_val

    # Add bias
    b_val = tl.load(in_b_ptr + m, mask=mask_m, other=0.0)
    acc += b_val

    # Store to out[b, s, m]
    out_idx = b * (S * M) + s * M + m
    tl.store(out_ptr + out_idx, acc, mask=mask_m)


# 2) gating_kernel: out[b, s, h] = B[b, s, h] * x_proj[b, s, h]
@triton.jit
def gating_kernel(
    B_ptr,        # *float32, (B, S, H)
    Xptr,         # *float32, (B, S, H) = x_proj
    out_ptr,      # *float32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h < H

    b_val = tl.load(B_ptr + b * (S * H) + s * H + h, mask=mask_h, other=0.0)
    x_val = tl.load(Xptr + b * (S * H) + s * H + h, mask=mask_h, other=0.0)
    out_val = b_val * x_val
    out_idx = b * (S * H) + s * H + h
    tl.store(out_ptr + out_idx, out_val, mask=mask_h)


# 3) left_pad_kernel: pad along sequence dimension by pad_left positions on the left
@triton.jit
def left_pad_kernel(
    inp_ptr,          # *float32, (B, S, H) = Bx
    out_ptr,          # *float32, (B, S_padded, H)
    B: tl.constexpr,
    S: tl.constexpr,          # original seq_len
    H: tl.constexpr,
    pad_left: tl.constexpr,   # int = K-1, e.g., 3
    S_padded: tl.constexpr,   # S + pad_left
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0..S_padded-1]
    h = tl.program_id(2)  # h in [0..H-1]
    # If t < pad_left: write 0; else copy inp[b, t - pad_left, h]
    if t < pad_left:
        tl.store(out_ptr + b * (S_padded * H) + t * H + h, 0.0)
    else:
        val = tl.load(inp_ptr + b * (S * H) + (t - pad_left) * H + h)
        tl.store(out_ptr + b * (S_padded * H) + t * H + h, val)


# 4) conv_groupsH_kernel: grouped conv with groups=B, per (b,c) conv across S
#    conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t + k] * conv_weight[c, 0, k] + conv_bias[c]
@triton.jit
def conv_groupsH_kernel(
    inp_ptr,   # *float32, (B, S_padded, H) = Bx_padded
    w_ptr,     # *float32, (H,) per channel flattened (since weight is (H,1,4))
    bias_ptr,  # *float32, (H,)
    out_ptr,   # *float32, (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S_padded: tl.constexpr,
    K: tl.constexpr,  # kernel_size, e.g., 4
):
    b = tl.program_id(0)
    c = tl.program_id(1)  # channel index in [0..H-1]
    # For each t in [0..S-1], compute conv_out[b, c, t]
    for t in range(0, S):
        acc = 0.0
        # sum over k=0..K-1
        for k in range(0, K):
            idx = b * (S_padded * H) + (t + k) * H + c
            acc += tl.load(inp_ptr + idx)
        # add bias
        bias_c = tl.load(bias_ptr + c)
        acc += bias_c
        # store to out[b, c, t]
        out_idx = b * (H * S) + c * S + t
        tl.store(out_ptr + out_idx, acc)


# 5) out_proj_kernel: final linear of y_T (B,S,H) -> output (B,S,H), W: (H,H), b: (H,)
@triton.jit
def out_proj_kernel(
    yT_ptr,     # *float32, (B, S, H) = y_T
    W_ptr,      # *float32, (H, H)
    b_ptr,      # *float32, (H,)
    out_ptr,    # *float32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h < H

    # Compute out[b, s, h] = sum_{h2} yT[b, s, h2] * W[h2, h] + b[h]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h2 in range(0, H):
        y = tl.load(yT_ptr + b * (S * H) + s * H + h2, mask=mask_h, other=0.0)
        w_vec = tl.load(W_ptr + h2 * H + h, mask=mask_h, other=0.0)
        acc += y * w_vec
    b_vec = tl.load(b_ptr + h, mask=mask_h, other=0.0)
    acc += b_vec

    out_idx = b * (S * H) + s * H + h
    tl.store(out_ptr + out_idx, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters in this module; we expect inputs with parameters passed at forward.
        # The original run(...) passes in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias
        # as external arguments. We'll still define a forward that takes them.

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor
                ) -> torch.Tensor:
        """
        Compute:
        1) BCx = x @ in_proj_weight^T + in_proj_bias  -> (B, S, 3H)
        2) B = BCx[:, :, :H], C = BCx[:, :, H:2H], x_proj = BCx[:, :, 2H:3H]
        3) Bx = B * x_proj
        4) Bx_padded = left-pad by K-1 on left
        5) conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t + k] * conv_weight[c, 0, k] + conv_bias[c]
           -> (B, H, S)
        6) y = C * conv_out
        7) output = linear(y_T) with out_proj_weight, out_proj_bias -> (B, S, H)
        """
        device = x.device
        dtype = x.dtype

        # Ensure float32 for consistent Triton compute
        x_f = x.contiguous().to(torch.float32)
        in_w_f = in_proj_weight.contiguous().to(torch.float32)  # (3H, H)
        in_b_f = in_proj_bias.contiguous().to(torch.float32)    # (3H,)
        conv_w_f = conv_weight.contiguous().to(torch.float32)   # (H, 1, 4)
        conv_b_f = conv_bias.contiguous().to(torch.float32)     # (H,)
        out_w_f = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_b_f = out_proj_bias.contiguous().to(torch.float32)    # (H,)

        B, S, H = x_f.shape
        M = 3 * H
        K = conv_w_f.shape[2]  # kernel size, e.g., 4
        pad_left = K - 1
        S_padded = S + pad_left

        # 1) in_proj: BCx (B, S, 3H)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)
        BLOCK_M = 64  # tile along M = 3H
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_f, in_w_f, in_b_f, BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj
        # BCx shape (B, S, 3H), split along last dim
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:(2 * H)]
        x_proj = BCx[:, :, (2 * H):]

        # 3) Element-wise gating: Bx = B_t * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        BLOCK_H = 64
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H))
        gating_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 4) Left-pad Bx by pad_left on left to get (B, S_padded, H)
        Bx_padded = torch.empty((B, S_padded, H), device=device, dtype=torch.float32)
        grid_pad = (B, S_padded, H)
        left_pad_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, S=S, H=H, pad_left=pad_left, S_padded=S_padded,
            num_warps=4, num_stages=2
        )

        # 5) Grouped depthwise conv along channels (groups=H): conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        # Flatten conv weight to (H,)
        w_flat = conv_w_f.reshape(H * K).contiguous()  # weight per channel
        # Launch per (b, c)
        grid_conv = (B, H)
        conv_groupsH_kernel[grid_conv](
            Bx_padded, w_flat, conv_b_f, conv_out,
            B=B, H=H, S_padded=S_padded, K=K,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out (B, H, S)
        y = C_t * conv_out  # elementwise multiply

        # 7) Transpose back to (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 8) Final linear projection (H,H) with bias (H,)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, out_w_f, out_b_f, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
