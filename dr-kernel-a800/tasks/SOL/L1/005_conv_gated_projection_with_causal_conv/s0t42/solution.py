import torch
import triton
import triton.language as tl

# Kernel 1: in_proj F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,), BCx: (B, S, 3H)
@triton.jit
def in_proj_kernel(
    x_ptr,                # *f32, shape (B, S, H)
    w_ptr,                # *f32, shape (3H, H)
    b_ptr,                # *f32, shape (3H,)
    bcx_ptr,              # *f32, shape (B, S, 3H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    m_block = tl.program_id(2)  # tile along 3H

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along output channel (3H) per (b, s)
    # mask for valid m_offsets
    mask_m = m_offsets < (3 * H)

    # accumulator for this (b, s) row
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over H dimension (input channels) - static loop
    for h_idx in range(0, H):
        # x[b, s, h_idx] is a scalar
        x_idx = b_id * (S * H) + s_id * H + h_idx
        x_val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
        # w[m_offsets, h_idx]: vector of length BLOCK_M
        w_idx = m_offsets * H + h_idx
        w_val = tl.load(w_ptr + w_idx, mask=mask_m, other=0.0)
        acc += w_val * x_val

    # add bias
    bias_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bias_vals

    # write BCx[b, s, m_offsets]
    bcx_idx = b_id * (S * (3 * H)) + s_id * (3 * H) + m_offsets
    tl.store(bcx_ptr + bcx_idx, acc, mask=mask_m)


# Kernel 2: element-wise gating Bx = B_t * x_proj
# B_t: (B, S, H), x_proj: (B, S, H), Bx: (B, S, H)
@triton.jit
def elementwise_gate_kernel(
    bt_ptr, xp_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H

    bt_idx = b_id * (S * H) + s_id * H + m_offsets
    xp_idx = b_id * (S * H) + s_id * H + m_offsets
    bt_vals = tl.load(bt_ptr + bt_idx, mask=mask_m, other=0.0)
    xp_vals = tl.load(xp_ptr + xp_idx, mask=mask_m, other=0.0)
    out_vals = bt_vals * xp_vals

    out_idx = b_id * (S * H) + s_id * H + m_offsets
    tl.store(out_ptr + out_idx, out_vals, mask=mask_m)


# Kernel 3: left-pad along sequence by pad_left (for causal conv)
# Bx: (B, S, H) -> Bx_padded: (B, S+pad_left, H)
@triton.jit
def pad_left_kernel(
    bx_ptr,               # *f32, shape (B, S, H)
    bxpad_ptr,            # *f32, shape (B, S+pad_left, H)
    pad_left: tl.constexpr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)  # t in [0, S+pad_left)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H

    # If t < pad_left, write zeros; else copy from Bx[b, t-pad_left, :]
    # Compute source index if valid
    src_t = t_id - pad_left
    valid_src = (t_id >= pad_left) & (src_t >= 0)
    # For valid positions, load from Bx; else use 0
    bx_idx = b_id * (S * H) + src_t * H + m_offsets
    # mask for valid (when src_t >= 0)
    bx_mask = mask_m & valid_src
    bx_vals = tl.load(bx_ptr + bx_idx, mask=bx_mask, other=0.0)

    out_idx = b_id * ((S + pad_left) * H) + t_id * H + m_offsets
    tl.store(bxpad_ptr + out_idx, bx_vals, mask=mask_m)


# Kernel 4: grouped 1D conv with groups=H (B, H, S_padded) -> (B, H, S)
# Input Bx_padded: (B, H, S_padded), conv_weight: (H, 4), conv_bias: (H,)
# For each (b, c, t): conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t + k] * conv_weight[c, k] + conv_bias[c]
@triton.jit
def conv1d_groupsH_kernel(
    bxpad_ptr,            # *f32, shape (B, H, S_padded)
    w_ptr,                # *f32, shape (H, 4) conv_weight per channel
    b_ptr,                # *f32, shape (H,) conv_bias
    conv_out_ptr,         # *f32, shape (B, H, S)
    B: tl.constexpr, H: tl.constexpr, S_padded: tl.constexpr, BLOCK_T: tl.constexpr, K: tl.constexpr
):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)  # channel index in [0, H)
    t_block = tl.program_id(2)

    t_start = t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S

    # bias for this channel
    bias_c = tl.load(b_ptr + c_id)

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # static loop over kernel taps K=4
    for k in range(0, K):
        src_t = t_offsets + k
        in_bounds = (src_t < S_padded) & mask_t
        # Load input values Bx_padded[b, c_id, src_t]
        bx_idx = b_id * (H * S_padded) + c_id * S_padded + src_t
        bx_vals = tl.load(bx_ptr + bx_idx, mask=in_bounds, other=0.0)
        # Load weight for this channel and tap
        w_val = tl.load(w_ptr + c_id * K + k)  # scalar
        acc += bx_vals * w_val

    # add bias
    acc += bias_c

    # store conv_out[b, c, t_offsets]
    conv_idx = b_id * (H * S) + c_id * S + t_offsets
    tl.store(conv_out_ptr + conv_idx, acc, mask=mask_t)


# Kernel 5: output gating y = C_t * conv_out (B, H, S)
@triton.jit
def output_gate_kernel(
    ct_ptr, conv_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)  # t in [0, S)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H

    # Load C_t[b, t_id, m_offsets]
    ct_idx = b_id * (S * H) + t_id * H + m_offsets
    conv_idx = b_id * (H * S) + m_offsets * S + t_id
    ct_vals = tl.load(ct_ptr + ct_idx, mask=mask_m, other=0.0)
    conv_vals = tl.load(conv_ptr + conv_idx, mask=mask_m, other=0.0)
    out_vals = ct_vals * conv_vals

    # Write to out[b, t_id, m_offsets]
    out_idx = b_id * (S * H) + t_id * H + m_offsets
    tl.store(out_ptr + out_idx, out_vals, mask=mask_m)


# Kernel 6: final linear projection (B, S, H) using out_proj_weight (H, H), out_proj_bias (H,)
# We implement y_out[b, s, h] = sum_h2 out_proj_weight[h, h2] * y[b, s, h2] + out_proj_bias[h]
@triton.jit
def final_linear_kernel(
    y_ptr,                # *f32, shape (B, S, H)
    w_ptr,                # *f32, shape (H, H)
    b_ptr,                # *f32, shape (H,)
    out_ptr,              # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr
):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # accumulator for output vector of length BLOCK_H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # loop over h2 in H (static)
    for h2 in range(0, H):
        # y[b, s, h2] vector across h_offsets
        y_idx = b_id * (S * H) + s_id * H + h2 + h_offsets  # y[b, s, h2] across h_offsets
        # since h2 is scalar, we need to gather y[b, s, h2] for each h_offsets: fixed h2 across vector
        # Construct per-element index:
        # y_idx_vec = b_id*(S*H) + s_id*H + h2 + h_offsets
        y_vals = tl.load(y_ptr + y_idx, mask=mask_h, other=0.0)  # load for each h_offsets, h2 fixed
        # out_proj_weight[h_offsets, h2]: vector
        w_idx = h_offsets * H + h2
        w_vals = tl.load(w_ptr + w_idx, mask=mask_h, other=0.0)
        acc += y_vals * w_vals

    # add bias
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals

    # store output[b, s, h_offsets]
    out_idx = b_id * (S * H) + s_id * H + h_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the original run function.
        All computation performed inside Triton kernels launched by forward.
        """
        device = x.device
        dtype = torch.float32

        # Ensure all tensors are float32 and contiguous
        x = x.to(dtype).contiguous()
        in_proj_weight = in_proj_weight.to(dtype).contiguous()
        in_proj_bias = in_proj_bias.to(dtype).contiguous()
        conv_weight = conv_weight.to(dtype).contiguous()
        conv_bias = conv_bias.to(dtype).contiguous()
        out_proj_weight = out_proj_weight.to(dtype).contiguous()
        out_proj_bias = out_proj_bias.to(dtype).contiguous()

        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel_size, given as 4
        pad_left = K - 1  # 3

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = torch.empty((B, S, 3 * H), device=device, dtype=dtype)

        BLOCK_M_in = 64
        grid_in = (B, S, triton.cdiv(3 * H, BLOCK_M_in))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_in,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B_t, C_t, x_proj along dim=1 (channels), each size H
        # Since Triton kernels cannot slice tensors, we do this in torch (metadata), and feed pointers accordingly.
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 3) Element-wise gating: Bx = B_t * x_proj (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_M_gate = 128
        grid_gate = (B, S, triton.cdiv(H, BLOCK_M_gate))
        elementwise_gate_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_gate,
            num_warps=4, num_stages=2
        )

        # 4) Pad left by pad_left to make causal conv valid
        Bx_padded = torch.empty((B, S + pad_left, H), device=device, dtype=dtype)

        BLOCK_M_pad = 128
        grid_pad = (B, S + pad_left, triton.cdiv(H, BLOCK_M_pad))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded, pad_left,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_pad,
            num_warps=4, num_stages=2
        )

        # 5) Grouped 1D convolution with groups=H (conv_weight: (H, 1, 4) -> (H, 4), conv_bias: (H,))
        conv_weight_K = conv_weight.reshape(H, K).contiguous()  # (H, 4)
        conv_bias_c = conv_bias.contiguous()  # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)

        BLOCK_T_conv = 64
        grid_conv = (B, H, triton.cdiv(S, BLOCK_T_conv))
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_K, conv_bias_c, conv_out,
            B=B, H=H, S_padded=S + pad_left, BLOCK_T=BLOCK_T_conv, K=K,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)

        BLOCK_M_out = 128
        grid_out = (B, S, triton.cdiv(H, BLOCK_M_out))
        output_gate_kernel[grid_out](
            C_t, conv_out, y,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_out,
            num_warps=4, num_stages=2
        )

        # 7) Final linear projection: y -> output using out_proj_weight (H, H), out_proj_bias (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H_fin = 64
        grid_fin = (B, S, triton.cdiv(H, BLOCK_H_fin))
        final_linear_kernel[grid_fin](
            y, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_fin,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
