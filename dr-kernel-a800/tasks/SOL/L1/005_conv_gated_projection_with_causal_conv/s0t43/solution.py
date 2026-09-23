import torch
import triton
import triton.language as tl

# Kernel 1: in_proj F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,)
# Output: BCx: (B, S, 3H)
@triton.jit
def in_proj_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < (3 * H)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # x[b, s, :] -> flattened as vector over H
    for h2 in range(0, H):
        # Load x[b, s, h2] across BLOCK_H output channels
        x_idx = b_id * (S * H) + s_id * H + h2  # fixed s and h2, vector over h_offsets
        # Pointer arithmetic for x is: x[b, s, h2] exists but we need x[b, s, h_offsets] when multiplying with w[h_offsets, h2]
        # Instead, load x[b, s, h_offsets] by broadcasting s and h_offsets:
        x_idx_vec = b_id * (S * H) + s_id * H + h_offsets  # since each h_offsets corresponds to a different channel in output
        # Correct way: x[b, s, h_offsets] means b*(S*H) + s*H + h_offsets
        x_vals = tl.load(x_ptr + x_idx_vec, mask=mask_h, other=0.0)
        w_idx = h_offsets * H + h2  # in_proj_weight[h_offsets, h2]
        w_vals = tl.load(w_ptr + w_idx, mask=mask_h, other=0.0)
        acc += x_vals * w_vals

    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals

    out_idx = b_id * (S * (3 * H)) + s_id * (3 * H) + h_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_h)


# Kernel 2: elementwise gating Bx = B_t * x_proj
# B_t: (B, S, H), x_proj: (B, S, H), Output: Bx: (B, S, H)
@triton.jit
def gating_kernel(B_t_ptr, x_proj_ptr, out_ptr,
                   B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                   BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    b_t_idx = b_id * (S * H) + s_id * H + h_offsets
    x_proj_idx = b_id * (S * H) + s_id * H + h_offsets
    b_t = tl.load(B_t_ptr + b_t_idx, mask=mask_h, other=0.0)
    x_proj = tl.load(x_proj_ptr + x_proj_idx, mask=mask_h, other=0.0)
    out = b_t * x_proj

    out_idx = b_id * (S * H) + s_id * H + h_offsets
    tl.store(out_ptr + out_idx, out, mask=mask_h)


# Kernel 3: left-pad along sequence by pad_left on the left
# Bx: (B, S, H), Output: Bx_padded: (B, S + pad_left, H)
@triton.jit
def pad_left_kernel(Bx_ptr, out_ptr, pad_left: tl.constexpr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # If t < pad_left, write zeros; else copy Bx[b, t - pad_left, :]
    t = t_id
    src_t = t - pad_left

    zeros = tl.zeros([BLOCK_H], dtype=tl.float32)
    is_pad = t < pad_left

    b_t_idx = b_id * (S * H) + src_t * H + h_offsets
    out_idx = b_id * ((S + pad_left) * H) + t * H + h_offsets

    if is_pad:
        tl.store(out_ptr + out_idx, zeros, mask=mask_h)
    else:
        vals = tl.load(Bx_ptr + b_t_idx, mask=mask_h, other=0.0)
        tl.store(out_ptr + out_idx, vals, mask=mask_h)


# Kernel 4: grouped 1D conv with groups=H (input: (B, H, S_padded), weight: (H, 1, 4), bias: (H,))
# Compute conv_out: (B, H, S_padded)
@triton.jit
def conv1d_groupsH_kernel(Bx_padded_ptr, w_ptr, bias_ptr, out_ptr,
                           B: tl.constexpr, H: tl.constexpr, S_padded: tl.constexpr,
                           BLOCK_T: tl.constexpr, K: tl.constexpr):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)
    t_block = tl.program_id(2)

    t_start = t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S_padded

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Static loop over K taps
    for k in range(0, K):
        vals = tl.load(Bx_padded_ptr + b_id * (H * S_padded) + c_id * S_padded + (t_offsets + k) * H, mask=mask_t, other=0.0)
        w_val = tl.load(w_ptr + c_id * K + k)  # weight[c, 0, k]
        acc += vals * w_val

    bias_val = tl.load(bias_ptr + c_id)
    acc += bias_val

    out_idx = b_id * (H * S_padded) + c_id * S_padded + t_offsets * H
    tl.store(out_ptr + out_idx, acc, mask=mask_t)


# Kernel 5: elementwise output gating y = C_t * conv_out
# C_t: (B, S, H), conv_out: (B, H, S) but we only use first S positions
@triton.jit
def gating_out_kernel(C_t_ptr, conv_out_ptr, out_ptr,
                      B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                      BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    C_t_idx = b_id * (S * H) + t_id * H + h_offsets
    conv_out_idx = b_id * (H * S) + h_offsets * S + t_id  # conv_out[b, h, t]
    C_t = tl.load(C_t_ptr + C_t_idx, mask=mask_h, other=0.0)
    conv_out = tl.load(conv_out_ptr + conv_out_idx, mask=mask_h, other=0.0)

    y = C_t * conv_out

    out_idx = b_id * (S * H) + t_id * H + h_offsets
    tl.store(out_ptr + out_idx, y, mask=mask_h)


# Kernel 6: final linear projection y -> output using out_proj_weight (H, H), out_proj_bias (H,)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def out_proj_kernel(y_ptr, w_ptr, b_ptr, out_ptr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for h2 in range(0, H):
        y_idx = b_id * (S * H) + s_id * H + h_offsets
        y_vals = tl.load(y_ptr + y_idx, mask=mask_h, other=0.0)
        w_idx = h_offsets * H + h2  # out_proj_weight[h_offsets, h2]
        w_vals = tl.load(w_ptr + w_idx, mask=mask_h, other=0.0)
        acc += y_vals * w_vals

    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

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
        K = conv_weight.shape[2]  # kernel_size (given as 4)
        pad_left = K - 1  # 3
        S_padded = S + pad_left

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias, output (B, S, 3H)
        BCx = torch.empty((B, S, 3 * H), device=device, dtype=dtype)
        BLOCK_H_in = 64
        grid_in = (B, S, triton.cdiv(3 * H, BLOCK_H_in))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_in,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B_t, C_t, x_proj along dim=1 (channels), each size H
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 3) Element-wise gating: Bx = B_t * x_proj, shape (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H_gate = 128
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H_gate))
        gating_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_gate,
            num_warps=4, num_stages=2
        )

        # 4) Pad left by pad_left to make causal conv valid
        Bx_padded = torch.empty((B, S_padded, H), device=device, dtype=dtype)
        BLOCK_H_pad = 128
        grid_pad = (B, S_padded, triton.cdiv(H, BLOCK_H_pad))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded, pad_left,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_pad,
            num_warps=4, num_stages=2
        )

        # 5) Grouped 1D convolution with groups=H
        # conv_weight: (H, 1, 4), conv_bias: (H,)
        conv_weight_k = conv_weight.reshape(H, K).contiguous()  # (H, 4)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)  # we will compute (B, H, S_padded) then take first S
        BLOCK_T_conv = 64
        grid_conv = (B, H, triton.cdiv(S_padded, BLOCK_T_conv))
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_k, conv_bias, conv_out,
            B=B, H=H, S_padded=S_padded, BLOCK_T=BLOCK_T_conv, K=K,
            num_warps=4, num_stages=2
        )

        # Note: conv_out is (B, H, S_padded); we need y = C_t * conv_out[:, :, :S]
        conv_out_S = conv_out[:, :, :S]  # (B, H, S)
        y = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H_gating_out = 128
        grid_gating_out = (B, S, triton.cdiv(H, BLOCK_H_gating_out))
        gating_out_kernel[grid_gating_out](
            C_t, conv_out_S, y,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_gating_out,
            num_warps=4, num_stages=2
        )

        # 6) Final linear projection: y -> output using out_proj_weight (H, H), out_proj_bias (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
