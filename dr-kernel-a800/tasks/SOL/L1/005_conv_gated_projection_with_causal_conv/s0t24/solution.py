import torch
import triton
import triton.language as tl

# Kernel 1: in_proj F.linear for x -> (B, S, 3H)
# Computes BCx[b, m, s] = sum_h x[b, s, h] * in_proj_weight[m, h] + in_proj_bias[m]
@triton.jit
def in_proj_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                    B, S, H, M,
                    stride_x_b, stride_x_s, stride_x_h,
                    stride_w_m, stride_w_h,
                    stride_out_b, stride_out_m, stride_out_s,
                    BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence index
    pid_m = tl.program_id(2)  # output channel index tile

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over hidden dimension h (static loop)
    for h in range(0, H):
        # Load x[b, s, h]
        x_val = tl.load(
            x_ptr + pid_b * stride_x_b + pid_s * stride_x_s + h * stride_x_h,
            mask=True, other=0.0
        )
        # Load in_proj_weight[m, h] for m_offsets
        w_ptr_h = w_ptr + m_offsets * stride_w_m + h * stride_w_h
        w_vals = tl.load(w_ptr_h, mask=mask_m, other=0.0)
        # Accumulate
        acc += x_val * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # Store BCx[b, m, s]
    out_ptr_tile = out_ptr + pid_b * stride_out_b + m_offsets * stride_out_m + pid_s * stride_out_s
    tl.store(out_ptr_tile, acc, mask=mask_m)

# Kernel 2: element-wise gating: Bx = B * x_proj
@triton.jit
def gate_kernel(B_ptr, x_ptr, out_ptr,
                B, S, H,
                stride_B_b, stride_B_s, stride_B_h,
                stride_x_b, stride_x_s, stride_x_h,
                stride_out_b, stride_out_s, stride_out_h,
                BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    B_vals = tl.load(B_ptr + pid_b * stride_B_b + pid_s * stride_B_s + h_offsets * stride_B_h,
                     mask=mask_h, other=0.0)
    x_vals = tl.load(x_ptr + pid_b * stride_x_b + pid_s * stride_x_s + h_offsets * stride_x_h,
                     mask=mask_h, other=0.0)
    out_vals = B_vals * x_vals

    tl.store(out_ptr + pid_b * stride_out_b + pid_s * stride_out_s + h_offsets * stride_out_h,
             out_vals, mask=mask_h)

# Kernel 3: left-pad along sequence by pad_left
@triton.jit
def pad_left_kernel(inp_ptr, out_ptr,
                     B, S, H, PAD,
                     stride_inp_b, stride_inp_s, stride_inp_h,
                     stride_out_b, stride_out_s, stride_out_h,
                     BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # For each out sequence index t in [0..S+PAD-1], if t < PAD -> 0, else inp[b, t-PAD, h]
    S_out = S + PAD
    for t in range(0, S_out):
        if t < PAD:
            # write zeros
            zeros = tl.zeros([BLOCK_H], dtype=tl.float32)
            tl.store(out_ptr + pid_b * stride_out_b + t * stride_out_s + h_offsets * stride_out_h,
                     zeros, mask=mask_h)
        else:
            src_s = t - PAD
            vals = tl.load(inp_ptr + pid_b * stride_inp_b + src_s * stride_inp_s + h_offsets * stride_inp_h,
                           mask=mask_h, other=0.0)
            tl.store(out_ptr + pid_b * stride_out_b + t * stride_out_s + h_offsets * stride_out_h,
                     vals, mask=mask_h)

# Kernel 4: grouped causal 1D conv with groups=H, kernel_size=4
# Input: Bx_padded (B, 3H, S+PAD), Weight: (H, 1, 4), Bias: (H,)
# Output: conv_out (B, H, S)
@triton.jit
def conv1d_groupsH_kernel(inp_ptr, w_ptr, b_ptr, out_ptr,
                           B, S, H, PAD,
                           stride_inp_b, stride_inp_s, stride_inp_h,
                           stride_w_c, stride_w_k,
                           stride_out_b, stride_out_h, stride_out_s,
                           BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # channel index
    pid_s = tl.program_id(2)  # sequence tile

    s_start = pid_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Iterate taps k in [0..3] (static)
    for k in range(0, 4):
        inp_ptr_k = inp_ptr + pid_b * stride_inp_b + (s_offsets + k) * stride_inp_s + pid_c * stride_inp_h
        vals = tl.load(inp_ptr_k, mask=mask_s, other=0.0)
        w_val = tl.load(w_ptr + pid_c * stride_w_c + k * stride_w_k)
        acc += vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_c)
    acc += b_val

    # Store conv_out[b, c, s]
    out_ptr_tile = out_ptr + pid_b * stride_out_b + pid_c * stride_out_h + s_offsets * stride_out_s
    tl.store(out_ptr_tile, acc, mask=mask_s)

# Kernel 5: final linear projection y -> (B, S, H)
# y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H,)
# Output: (B, S, H)
@triton.jit
def out_proj_kernel(y_ptr, w_ptr, b_ptr, out_ptr,
                    B, S, H,
                    stride_y_b, stride_y_h, stride_y_s,
                    stride_w_h, stride_w_h2,
                    stride_out_b, stride_out_s, stride_out_h,
                    BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    s_start = pid_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Compute acc[BLOCK_S, BLOCK_H] = y[b, h2, s] dot w[h2, h]
    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)

    # Iterate over h2 (hidden dimension)
    for h2 in range(0, H):
        # Load y[b, h2, s]
        y_vals = tl.load(y_ptr + pid_b * stride_y_b + h2 * stride_y_h + s_offsets * stride_y_s,
                         mask=mask_s, other=0.0)  # [BLOCK_S]
        # Load w[h2, h_offsets]
        w_vals = tl.load(w_ptr + h2 * stride_w_h + h_offsets * stride_w_h2,
                         mask=mask_h, other=0.0)   # [BLOCK_H]
        # Outer product accumulate
        acc += y_vals[:, None] * w_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals[None, :]

    # Store out[b, s, h]
    out_ptr_tile = out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    mask = mask_s[:, None] & mask_h[None, :]
    tl.store(out_ptr_tile, acc, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H)
        in_proj_bias: (3H,)
        conv_weight: (H, 1, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        Output: (B, S, H)
        """
        device = x.device
        dtype = x.dtype
        B, S, H = x.shape
        M = 3 * H

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)
        # Strides
        stride_x_b, stride_x_s, stride_x_h = x.stride(0), x.stride(1), x.stride(2)
        stride_w_m, stride_w_h = in_proj_weight.stride(0), in_proj_weight.stride(1)
        stride_out_b, stride_out_m, stride_out_s = BCx.stride(0), BCx.stride(1), BCx.stride(2)

        # Launch Triton kernel
        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, M,
            stride_x_b, stride_x_s, stride_x_h,
            stride_w_m, stride_w_h,
            stride_out_b, stride_out_m, stride_out_s,
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Transpose to (B, 3H, S)
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, 3H, S)

        # 3) Split into B, C, x_proj along dim=1 (each shape (B, S, H))
        # This is done via views; since BCx_T is contiguous, chunking is safe.
        B_ = BCx_T[:, :H, :]               # (B, H, S)
        C_ = BCx_T[:, H:2*H, :]           # (B, H, S)
        x_proj = BCx_T[:, 2*H:3*H, :]     # (B, H, S)

        # Make sure they are contiguous (they are due to transpose)
        B_ = B_.contiguous()
        C_ = C_.contiguous()
        x_proj = x_proj.contiguous()

        # 4) Gating: Bx = B_ * x_proj (elementwise)
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        stride_B_b, stride_B_s, stride_B_h = B_.stride(0), B_.stride(1), B_.stride(2)
        stride_x_b, stride_x_s, stride_x_h = x_proj.stride(0), x_proj.stride(1), x_proj.stride(2)
        stride_out_b, stride_out_s, stride_out_h = Bx.stride(0), Bx.stride(1), Bx.stride(2)

        BLOCK_H = 64
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H))
        gate_kernel[grid_gate](
            B_, x_proj, Bx,
            B, S, H,
            stride_B_b, stride_B_s, stride_B_h,
            stride_x_b, stride_x_s, stride_x_h,
            stride_out_b, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 5) Left-pad along sequence by PAD=K-1=3 to make S+PAD
        PAD = conv_weight.shape[2] - 1  # K=4 -> PAD=3
        S_padded = S + PAD
        Bx_padded = torch.empty((B, 3*H, S_padded), device=device, dtype=dtype)

        stride_inp_b, stride_inp_s, stride_inp_h = Bx.stride(0), Bx.stride(1), Bx.stride(2)
        stride_out_b, stride_out_s, stride_out_h = Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2)

        BLOCK_H_pad = 64
        grid_pad = (B, S_padded, triton.cdiv(3*H, BLOCK_H_pad))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded,
            B, S, 3*H, PAD,
            stride_inp_b, stride_inp_s, stride_inp_h,
            stride_out_b, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H_pad,
            num_warps=4, num_stages=2
        )

        # 6) Grouped causal conv: groups=H, kernel_size=4
        # conv_weight: (H, 1, 4) -> (H, 4), conv_bias: (H,)
        conv_weight_ = conv_weight[:, 0, :].contiguous()  # (H, 4)
        conv_bias_ = conv_bias.contiguous()               # (H,)

        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)

        stride_inp_b, stride_inp_s, stride_inp_h = Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2)
        stride_w_c, stride_w_k = conv_weight_.stride(0), conv_weight_.stride(1)
        stride_out_b, stride_out_h, stride_out_s = conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)

        BLOCK_H_conv = 64
        BLOCK_S_conv = 128
        grid_conv = (B, H, triton.cdiv(S, BLOCK_S_conv))
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_, conv_bias_, conv_out,
            B, S, H, PAD,
            stride_inp_b, stride_inp_s, stride_inp_h,
            stride_w_c, stride_w_k,
            stride_out_b, stride_out_h, stride_out_s,
            BLOCK_H=BLOCK_H_conv, BLOCK_S=BLOCK_S_conv,
            num_warps=4, num_stages=2
        )

        # 7) Output gating: y = C_ * conv_out (elementwise), conv_out: (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        # y = C_ * conv_out
        y = C_ * conv_out  # elementwise multiply

        # 8) Transpose back to (B, S, H) for final linear projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 9) Final linear projection using Triton
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        out_proj_weight_c = out_proj_weight.contiguous()     # (H, H)
        out_proj_bias_c = out_proj_bias.contiguous()         # (H,)

        stride_y_b, stride_y_h, stride_y_s = y_T.stride(0), y_T.stride(1), y_T.stride(2)
        stride_w_h, stride_w_h2 = out_proj_weight_c.stride(0), out_proj_weight_c.stride(1)
        stride_out_b, stride_out_s, stride_out_h = output.stride(0), output.stride(1), output.stride(2)

        BLOCK_H_out = 64
        BLOCK_S_out = 128
        grid_out = (B, triton.cdiv(S, BLOCK_S_out), triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight_c, out_proj_bias_c, output,
            B, S, H,
            stride_y_b, stride_y_h, stride_y_s,
            stride_w_h, stride_w_h2,
            stride_out_b, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H_out, BLOCK_S=BLOCK_S_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
