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
    # Grid: (B, S, tiles along M where M=3*H)
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    m_block = tl.program_id(2)

    M = 3 * H
    m_start = m_block * BLOCK_H
    m_offsets = m_start + tl.arange(0, BLOCK_H)
    mask_m = m_offsets < M

    # Accumulator for outputs of size BLOCK_H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over H-dimension for dot product: y[b, s, m] = sum_h x[b, s, h] * w[m, h] + bias[m]
    for h in range(0, H):
        # Load x[b, s, h] for m_offsets
        x_idx = b_id * (S * H) + s_id * H + h
        x_vals = tl.load(x_ptr + x_idx, mask=mask_m, other=0.0)  # shape [BLOCK_H]
        # Load w[m_offsets, h], w is (3H, H)
        w_idx = m_offsets * H + h
        w_vals = tl.load(w_ptr + w_idx, mask=mask_m, other=0.0)  # shape [BLOCK_H]
        acc += x_vals * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bias_vals

    # Store BCx[b, s, m_offsets]
    out_idx = b_id * (S * M) + s_id * M + m_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_m)


# Kernel 2: Element-wise gating: Bx = B_t * x_proj, (B, S, H)
@triton.jit
def gate_kernel(B_t_ptr, x_proj_ptr, out_ptr,
                B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    B_vals = tl.load(B_t_ptr + (b_id * S + s_id) * H + h_offsets, mask=mask_h, other=0.0)
    X_vals = tl.load(x_proj_ptr + (b_id * S + s_id) * H + h_offsets, mask=mask_h, other=0.0)
    out_vals = B_vals * X_vals

    tl.store(out_ptr + (b_id * S + s_id) * H + h_offsets, out_vals, mask=mask_h)


# Kernel 3: Left-pad along sequence by pad_left (K-1), input (B, H, S) -> output (B, H, S+pad_left)
@triton.jit
def pad_left_kernel(inp_ptr, out_ptr, pad_left: tl.constexpr,
                    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    h_block = tl.program_id(1)
    s_out = tl.program_id(2)  # s_out in [0, S+pad_left)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # If s_out < pad_left: pad zeros; else copy from inp[b, :, s_out - pad_left]
    in_s = s_out - pad_left
    is_valid = in_s >= 0
    inp_idx = b_id * (H * S) + in_s * H + h_offsets
    out_idx = b_id * (H * (S + pad_left)) + s_out * H + h_offsets

    if is_valid:
        vals = tl.load(inp_ptr + inp_idx, mask=mask_h, other=0.0)
    else:
        vals = tl.zeros([BLOCK_H], dtype=tl.float32)

    tl.store(out_ptr + out_idx, vals, mask=mask_h)


# Kernel 4: Grouped 1D convolution with groups=H.
# Input: Bx_padded (B, H, S+pad_left), conv_weight: (H, 1, 4), conv_bias: (H,)
# Output: conv_out (B, H, S)
@triton.jit
def conv1d_groupsH_kernel(inp_ptr, w_ptr, b_ptr, out_ptr,
                           B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                           pad_left: tl.constexpr,
                           BLOCK_S: tl.constexpr):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)
    s_block = tl.program_id(2)

    S_padded = S + pad_left
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for each s_offsets
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Static loop over K=4 taps for grouped conv with groups=H
    # For each output time t in s_offsets, sum over k in [0..3] of inp[b, c, t + k] * w[c, 0, k] + b[c]
    # Note: w_ptr indexed by (c, k), as conv_weight is (H, 1, 4) but we pass as (H, 4) contiguous.
    for k in range(0, 4):
        t_in = s_offsets + k
        in_bounds = t_in < S_padded
        inp_idx = b_id * (H * S_padded) + c_id * S_padded + t_in
        inp_vals = tl.load(inp_ptr + inp_idx, mask=in_bounds, other=0.0)  # [BLOCK_S]
        w_val = tl.load(w_ptr + c_id * 4 + k)  # single scalar per c,k
        acc += inp_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + c_id)
    acc += b_val

    # Store to conv_out[b, c, s_offsets]
    out_idx = b_id * (H * S) + c_id * S + s_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_s)


# Kernel 5: Output gating: y = C_t * conv_out, elementwise, (B, H, S)
@triton.jit
def out_gate_kernel(C_t_ptr, conv_out_ptr, out_ptr,
                    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    h_block = tl.program_id(1)
    s_id = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    C_vals = tl.load(C_t_ptr + (b_id * H + s_id) * H + h_offsets, mask=mask_h, other=0.0)
    Conv_vals = tl.load(conv_out_ptr + (b_id * H + s_id) * H + h_offsets, mask=mask_h, other=0.0)
    out_vals = C_vals * Conv_vals

    tl.store(out_ptr + (b_id * H + s_id) * H + h_offsets, out_vals, mask=mask_h)


# Kernel 6: Final linear projection to output (B, S, H), using out_proj_weight (H, H), out_proj_bias (H,)
# y_T: (B, S, H) where y_T[b, s, h] is input to dot-product for output[b, s, h]
@triton.jit
def out_proj_kernel(y_T_ptr, out_w_ptr, out_b_ptr, out_ptr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Accumulator for output[b, s, h_offsets]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # For each h2 in H, sum y_T[b, s, h2] * out_w[h2, h_offsets]
    # y_T: (B, S, H) linear index = b*(S*H) + s*H + h
    for h2 in range(0, H):
        y_idx = b_id * (S * H) + s_id * H + h2
        y_val = tl.load(y_T_ptr + y_idx, mask=mask_h, other=0.0)  # scalar for each h_offsets?
        # Note: We need dot over H. To do that, we need y_T over h dimension. Implement a 2D loop over h2 and h_offsets.
        # Better approach: compute acc[h_offsets] += sum_h2 y_T[b, s, h2] * out_w[h2, h_offsets]
        # Since h_offsets is a vector, we load out_w[h2, h_offsets] vector and multiply with y_val vector (broadcasted).
        # But Triton doesn't support vectorized broadcasting like numpy; we must compute scalar-wise and store per h.
        # Instead, we restructure as: out_w_ptr is (H, H) contiguous; out_w[h2, h_offsets] loads a vector for each h2.
        out_w_vals = tl.load(out_w_ptr + h2 * H + h_offsets, mask=mask_h, other=0.0)
        # Here y_val is scalar; multiply with vector and accumulate:
        acc += y_val * out_w_vals

    # Add bias
    b_vals = tl.load(out_b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    # Store output[b, s, h_offsets]
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
        in_proj_weight = in_proj_weight.to(dtype).contiguous()  # (3H, H)
        in_proj_bias = in_proj_bias.to(dtype).contiguous()      # (3H,)
        conv_weight = conv_weight.to(dtype).contiguous()        # (H, 1, 4)
        conv_bias = conv_bias.to(dtype).contiguous()            # (H,)
        out_proj_weight = out_proj_weight.to(dtype).contiguous()  # (H, H)
        out_proj_bias = out_proj_bias.to(dtype).contiguous()      # (H,)

        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel_size, fixed at 4
        pad_left = K - 1  # 3

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = torch.empty((B, S, 3 * H), device=device, dtype=dtype)
        BLOCK_H_in = 64
        grid_in = (B, S, triton.cdiv(3 * H, BLOCK_H_in))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_in,
            num_warps=4, num_stages=2
        )

        # Split BCx into three groups along channels (dim=1)
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 2) Element-wise gating: Bx = B_t * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H_gate = 64
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H_gate))
        gate_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_gate,
            num_warps=4, num_stages=2
        )

        # 3) Left-pad for causal conv by pad_left = K-1 = 3
        Bx_padded = torch.empty((B, H, S + pad_left), device=device, dtype=dtype)
        BLOCK_H_pad = 64
        grid_pad = (B, triton.cdiv(H, BLOCK_H_pad), S + pad_left)
        pad_left_kernel[grid_pad](
            Bx, Bx_padded, pad_left,
            B=B, H=H, S=S, BLOCK_H=BLOCK_H_pad,
            num_warps=4, num_stages=2
        )

        # 4) Grouped 1D convolution with groups=H: conv_weight (H, 1, 4), conv_bias (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        BLOCK_S_conv = 64
        grid_conv = (B, H, triton.cdiv(S, BLOCK_S_conv))
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight.reshape(H, 4).contiguous(), conv_bias.contiguous(), conv_out,
            B=B, H=H, S=S, pad_left=pad_left, BLOCK_S=BLOCK_S_conv,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C_t * conv_out, elementwise (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        BLOCK_H_out = 64
        grid_out = (B, triton.cdiv(H, BLOCK_H_out), S)
        out_gate_kernel[grid_out](
            C_t, conv_out, y,
            B=B, H=H, S=S, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        # 6) Transpose y to (B, S, H) for final linear projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 7) Final linear projection: y_T -> output using out_proj_weight (H, H), out_proj_bias (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_lin = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_lin](
            y_T, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
