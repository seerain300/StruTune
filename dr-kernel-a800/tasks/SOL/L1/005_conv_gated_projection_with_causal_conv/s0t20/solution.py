import torch
import triton
import triton.language as tl

# Kernel 1: in_proj F.linear(x, W, b), produce BCx[b, m, s] where W: (M=3H, H), x: (B, S, H)
@triton.jit
def in_proj_kernel(
    x_ptr,             # *float32, shape (B, S, H)
    W_ptr,             # *float32, shape (M=3H, H)
    b_ptr,             # *float32, shape (M=3H,)
    BCx_ptr,           # *float32, shape (B, M, S) but we store BCx[b, m, s]
    B: tl.constexpr,   # int
    M: tl.constexpr,   # int, = 3*H
    S: tl.constexpr,   # int
    H: tl.constexpr,   # int
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    s = tl.program_id(2)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Accumulate over h from 0 to H-1
    for h in range(0, H):
        x_off = b * S * H + s * H + h  # x[b, s, h]
        x_val = tl.load(x_ptr + x_off, mask=True, other=0.0)
        W_off = m_offsets * H + h
        W_val = tl.load(W_ptr + W_off, mask=mask_m, other=0.0)
        acc += x_val * W_val

    bIAS = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bIAS

    out_off = b * M * S + m_offsets * S + s
    tl.store(BCx_ptr + out_off, acc, mask=mask_m)


# Kernel 2: element-wise gating Bx = B * x_proj
# BCx has shape (B, 3H, S) where B = BCx[:, :H, :], x_proj = BCx[:, 2H:3H, :]
@triton.jit
def gating_kernel(
    BCx_ptr,        # *float32, shape (B, 3H, S)
    Bx_ptr,         # *float32, shape (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h_block = tl.program_id(1)
    s_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_h = h_offsets < H
    mask_s = s_offsets < S

    B_off = b * H * S + h_offsets[:, None] * S + s_offsets[None, :]
    x_off = b * H * S + (H + h_offsets[:, None]) * S + s_offsets[None, :]

    B_val = tl.load(BCx_ptr + B_off, mask=mask_h[:, None] & mask_s[None, :], other=0.0)
    x_val = tl.load(BCx_ptr + x_off, mask=mask_h[:, None] & mask_s[None, :], other=0.0)
    res = B_val * x_val
    out_off = b * H * S + h_offsets[:, None] * S + s_offsets[None, :]
    tl.store(Bx_ptr + out_off, res, mask=mask_h[:, None] & mask_s[None, :])


# Kernel 3: left-pad along sequence dimension for causal conv: Bx_padded[b, m, t] = Bx[b, m, t] if t >= pad else 0
@triton.jit
def pad_left_kernel(
    Bx_ptr,          # *float32, shape (B, 3H, S)
    Bx_padded_ptr,   # *float32, shape (B, 3H, S_padded)
    B: tl.constexpr,
    M: tl.constexpr,       # 3H
    S: tl.constexpr,
    pad_left: tl.constexpr,
    S_padded: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    t_block = tl.program_id(2)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    t_offsets = t_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_m = m_offsets < M
    mask_t = t_offsets < S_padded

    # For each t, if t < pad_left, write 0; else write Bx[b, m, t-pad_left]
    is_pad = t_offsets < pad_left
    # Compute source t_idx
    src_t = t_offsets - pad_left
    # Valid if not pad and src_t in [0, S-1]
    valid = (~is_pad) & (src_t >= 0) & (src_t < S) & mask_t

    # Load from Bx where valid; otherwise 0
    src_off = b * M * S + m_offsets[:, None] * S + src_t[None, :]
    val = tl.load(Bx_ptr + src_off, mask=valid, other=0.0)
    # For pad positions, val should be 0; for invalid t_offsets beyond S_padded, mask_t prevents writing
    out_off = b * M * S_padded + m_offsets[:, None] * S_padded + t_offsets[None, :]
    tl.store(Bx_padded_ptr + out_off, val, mask=mask_m[:, None] & mask_t[None, :])


# Kernel 4: grouped causal 1D convolution with groups=B
# Input: Bx_padded: (B, 3H, S_padded), weight: (H, 1, 4), bias: (H,)
# Output: conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t+k] * weight[c, 0, k] + bias[c]
@triton.jit
def conv1d_groupsB_kernel(
    Bx_padded_ptr,    # *float32, (B, 3H, S_padded)
    W_ptr,            # *float32, (H, 4), conv_weight per channel
    bIAS_ptr,         # *float32, (H,)
    conv_out_ptr,     # *float32, (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S_padded: tl.constexpr,
    K: tl.constexpr,     # kernel_size, e.g., 4
    BLOCK_T: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    t_block = tl.program_id(2)
    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S

    # Accumulate over K taps
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    # Weight vector for channel c
    w = tl.load(W_ptr + c * K + tl.arange(0, K), mask=True, other=0.0).to(tl.float32)

    # For each k in [0..K-1], add Bx_padded[b, c, t+k] * w[k]
    for k in range(0, K):
        src_t = t_offsets + k
        in_bounds = (src_t < S_padded) & mask_t
        val = tl.load(Bx_padded_ptr + b * (3 * H) * S_padded + c * S_padded + src_t, mask=in_bounds, other=0.0)
        acc += val * w[k]

    # Add bias
    bias_val = tl.load(bIAS_ptr + c, mask=True, other=0.0)
    acc += bias_val

    # Store conv_out[b, c, t]
    out_off = b * H * S + c * S + t_offsets
    tl.store(conv_out_ptr + out_off, acc, mask=mask_t)


# Kernel 5: output gating y = C * conv_out, where C = BCx[:, H:2H, :]
@triton.jit
def gating_output_kernel(
    BCx_ptr,          # *float32, (B, 3H, S)
    conv_out_ptr,     # *float32, (B, H, S)
    y_ptr,            # *float32, (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h_block = tl.program_id(1)
    s_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_h = h_offsets < H
    mask_s = s_offsets < S

    # C = BCx[:, H:2H, :]
    C_off = b * (3 * H) * S + (H + h_offsets[:, None]) * S + s_offsets[None, :]
    conv_off = b * H * S + h_offsets[:, None] * S + s_offsets[None, :]

    C_val = tl.load(BCx_ptr + C_off, mask=mask_h[:, None] & mask_s[None, :], other=0.0)
    conv_val = tl.load(conv_out_ptr + conv_off, mask=mask_h[:, None] & mask_s[None, :], other=0.0)
    res = C_val * conv_val

    y_off = b * H * S + h_offsets[:, None] * S + s_offsets[None, :]
    tl.store(y_ptr + y_off, res, mask=mask_h[:, None] & mask_s[None, :])


# Kernel 6: final linear (out_proj), y_T: (B, S, H), W_out: (H, H), b_out: (H,)
# output[b, s, h] = sum_{h2} y_T[b, s, h2] * W_out[h, h2] + b_out[h]
@triton.jit
def out_proj_kernel(
    y_T_ptr,          # *float32, (B, S, H)
    W_ptr,            # *float32, (H, H)
    b_ptr,            # *float32, (H,)
    out_ptr,          # *float32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # For each output h, accumulate over h2 in H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    # y_T[b, s, h2] for h2 in [0..H-1]
    for h2 in range(0, H):
        y_val = tl.load(y_T_ptr + b * S * H + s * H + h2, mask=True, other=0.0)
        # W[h, h2]
        W_val = tl.load(W_ptr + h_offsets * H + h2, mask=mask_h, other=0.0)
        acc += y_val * W_val

    bIAS = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bIAS

    out_off = b * S * H + s * H + h_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias,
                out_proj_weight, out_proj_bias):
        # Ensure dtype is float32 for correctness checks (original code uses default float32)
        device = x.device
        dtype = torch.float32
        x = x.contiguous().to(dtype)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)
        conv_weight = conv_weight.contiguous().to(dtype)
        conv_bias = conv_bias.contiguous().to(dtype)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)

        B, S, H = x.shape
        M = 3 * H
        K = conv_weight.shape[2]
        groups = H  # groups=hidden_size as per original

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = torch.empty((B, M, S), device=device, dtype=dtype)
        BLOCK_M = 64
        grid_in = (B, triton.cdiv(M, BLOCK_M), S)
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, M=M, S=S, H=H, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * x_proj
        # BCx: (B, 3H, S)
        Bx = torch.empty((B, H, S), device=device, dtype=dtype)
        BLOCK_H = 64
        BLOCK_S = 128
        grid_gate = (B, triton.cdiv(H, BLOCK_H), triton.cdiv(S, BLOCK_S))
        gating_kernel[grid_gate](
            BCx, Bx, B=B, H=H, S=S, BLOCK_H=BLOCK_H, BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 3) Left-pad Bx by pad_left = K - 1
        pad_left = K - 1
        S_padded = S + pad_left
        Bx_padded = torch.empty((B, 3 * H, S_padded), device=device, dtype=dtype)
        BLOCK_M_pad = 64
        BLOCK_S_pad = 128
        grid_pad = (B, triton.cdiv(3 * H, BLOCK_M_pad), triton.cdiv(S_padded, BLOCK_S_pad))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded, B=B, M=3 * H, S=S, pad_left=pad_left, S_padded=S_padded,
            BLOCK_M=BLOCK_M_pad, BLOCK_S=BLOCK_S_pad, num_warps=4, num_stages=2
        )

        # 4) Grouped causal conv: groups=B and per-channel weights (H, 1, 4)
        # conv_weight: (H, 1, 4) -> flatten to (H, 4)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        # Since we're doing grouped per (b, c) across the input's channel dim, we interpret conv_out[b, c, t]
        # as per the input after gating. We convolve each (b, c) slice on Bx_padded[b, c, :] and output length S.
        # Note: original code uses groups=hidden_size for conv1d over (B, H, S), here we interpret groups as per-channel.
        BLOCK_T = 128
        grid_conv = (B, H, triton.cdiv(S, BLOCK_T))
        conv1d_groupsB_kernel[grid_conv](
            Bx_padded, conv_weight.reshape(H, K).contiguous(), conv_bias.contiguous(),
            conv_out, B=B, H=H, S_padded=S_padded, K=K, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out, where C = BCx[:, H:2H, :] after the original transpose operation.
        # We need C corresponding to the original BCx shape (B, 3H, S). After transpose, C resides at channel index H..2H.
        # Here, we compute C from BCx (B, 3H, S) as BCx[:, H:2H, :], shape (B, H, S).
        C_t = BCx[:, H:(2 * H), :]  # (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_gating = (B, triton.cdiv(H, BLOCK_H), triton.cdiv(S, BLOCK_S))
        gating_output_kernel[grid_gating](
            BCx, conv_out, y, B=B, H=H, S=S, BLOCK_H=BLOCK_H, BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 6) Final linear projection to (B, S, H)
        # y_T: transpose y to (B, S, H)
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
