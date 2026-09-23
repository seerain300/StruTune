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
                    stride_out_b, stride_out_s, stride_out_m,
                    BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence index
    pid_m = tl.program_id(2)  # output channel tile index

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over hidden dimension h statically
    for h in range(0, H):
        # Load x[b, s, h]
        x_val = tl.load(
            x_ptr + pid_b * stride_x_b + pid_s * stride_x_s + h * stride_x_h,
            mask=True,
            other=0.0
        )
        # Load in_proj_weight[m, h] for all m in tile
        w_vals = tl.load(
            w_ptr + m_offsets * stride_w_m + h * stride_w_h,
            mask=mask_m,
            other=0.0
        )
        # Accumulate: acc[m] += x_val * w_vals[m]
        acc += x_val * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # Store to out[b, m, s]
    tl.store(out_ptr + pid_b * stride_out_b + m_offsets * stride_out_m + pid_s * stride_out_s, acc, mask=mask_m)

# Kernel 2: elementwise gating: Bx = B * x_proj
@triton.jit
def gate_kernel(B_ptr, x_proj_ptr, out_ptr,
                B, S, H,
                stride_B_b, stride_B_s, stride_B_h,
                stride_xp_b, stride_xp_s, stride_xp_h,
                stride_out_b, stride_out_s, stride_out_h,
                BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    B_vals = tl.load(B_ptr + pid_b * stride_B_b + pid_s * stride_B_s + h_offsets * stride_B_h, mask=mask, other=0.0)
    xP_vals = tl.load(x_proj_ptr + pid_b * stride_xp_b + pid_s * stride_xp_s + h_offsets * stride_xp_h, mask=mask, other=0.0)
    out_vals = B_vals * xP_vals
    tl.store(out_ptr + pid_b * stride_out_b + pid_s * stride_out_s + h_offsets * stride_out_h, out_vals, mask=mask)

# Kernel 3: left-pad along sequence by pad_left on the left (returning (B, T_in, H))
@triton.jit
def pad_left_kernel(inp_ptr, out_ptr,
                     B, T_in, H, pad_left,
                     stride_inp_b, stride_inp_t, stride_inp_h,
                     stride_out_b, stride_out_t, stride_out_h,
                     BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)  # t in [0, T_in)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # If t < pad_left: write zeros; else write inp[b, t - pad_left, :]
    write_mask = pid_t < pad_left
    src_t = pid_t - pad_left
    # Load from inp if not pad
    inp_vals = tl.load(inp_ptr + pid_b * stride_inp_b + src_t * stride_inp_t + h_offsets * stride_inp_h,
                       mask=mask_h & (~write_mask), other=0.0)
    # For pad positions, zeros
    out_vals = inp_vals + 0.0  # initialize
    out_vals = tl.where(write_mask, tl.zeros_like(out_vals), out_vals)

    # Store to out[b, t, :]
    tl.store(out_ptr + pid_b * stride_out_b + pid_t * stride_out_t + h_offsets * stride_out_h,
             out_vals, mask=mask_h)

# Kernel 4: grouped causal conv with groups=H, kernel_size=K
# Input: Bx_padded with shape (B, T_in, H), weight (H, 1, K), bias (H,)
# Output: conv_out (B, H, S)
@triton.jit
def conv1d_groupsH_kernel(inp_ptr, w_ptr, b_ptr, out_ptr,
                           B, T_in, H, K,
                           stride_inp_b, stride_inp_t, stride_inp_h,
                           stride_w_h, stride_w_k,
                           stride_out_b, stride_out_h, stride_out_s,
                           BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)  # output channel index tile
    pid_s = tl.program_id(2)  # output time index tile

    h_start = pid_h * BLOCK_H
    s_start = pid_s * BLOCK_S  # BLOCK_S is 1 for simplicity; we loop over s directly

    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # For each output time t in S
    # Note: We compute one s index per program; but we use a while-loop (supported in Triton) to iterate over S
    # We'll launch grid over (B, ceil(H/BLOCK_H), S). For each pid_s, we can only handle one s; so we loop.
    # To simplify, we use a 3D grid (B, H_tiles, S). Inside, we loop over s sequentially.
    # However Triton requires static loops; so we unroll per-launch with s being the grid's third dim.
    # Instead, we compute s vector as s_start + tl.arange(0, BLOCK_S). With BLOCK_S=1, this is fine.
    # We'll set BLOCK_S=1 to allow per-s iteration.
    s_vec = s_start + tl.arange(0, 1)  # only one element per program
    mask_s = s_vec < S

    # Accumulator for conv_out[b, h, s]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # For each k in K (static)
    for k in range(0, K):
        # For each h in H tile (static loop)
        # We need to load inp[b, s_vec, h_offsets + k] (causal index), valid if s_vec + k < T_in
        # Since s_vec is a single element, this is straightforward.
        for h_idx in range(0, BLOCK_H):
            h = h_offsets[h_idx]
            if h_idx >= H:
                continue
            # t = s_vec + k
            t = s_vec + k  # vectorized over s
            # Valid if t < S (output range), and inp index within T_in
            valid_t = t < S
            # Load inp[b, t, h]
            inp_val = tl.load(
                inp_ptr + pid_b * stride_inp_b + t * stride_inp_t + h * stride_inp_h,
                mask=mask_s & valid_t,
                other=0.0
            )
            # Load weight[h, 0, k]
            w_val = tl.load(w_ptr + h * stride_w_h + k * stride_w_k)
            acc[h_idx] += inp_val * w_val

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    # Store conv_out[b, h, s]
    tl.store(out_ptr + pid_b * stride_out_b + h_offsets * stride_out_h + s_vec * stride_out_s,
             acc, mask=mask_h & mask_s)

# Kernel 5: final linear projection y -> output (B, S, H)
# where y is (B, H, S), out_proj_weight is (H, H), out_proj_bias (H,)
@triton.jit
def out_proj_kernel(y_ptr, w_ptr, b_ptr, out_ptr,
                    B, S, H,
                    stride_y_b, stride_y_h, stride_y_s,
                    stride_w_h, stride_w_k,
                    stride_out_b, stride_out_s, stride_out_h,
                    BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Initialize output accumulator
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # For each h_out in the tile, compute dot over H
    for h_out_idx in range(0, BLOCK_H):
        h_out = h_offsets[h_out_idx]
        if h_out_idx >= H:
            continue
        # Load y[b, h_out, s] as vector over S (we'll use s=pid_s)
        y_val = tl.load(
            y_ptr + pid_b * stride_y_b + h_out * stride_y_h + pid_s * stride_y_s,
            mask=True,
            other=0.0
        )
        # Load out_proj_weight[h_out, :] over H (vector)
        w_vals = tl.load(
            w_ptr + h_out * stride_w_h + h_offsets * stride_w_k,
            mask=mask_h,
            other=0.0
        )
        acc[h_out_idx] = y_val * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    # Store output[b, s, h_offsets]
    tl.store(out_ptr + pid_b * stride_out_b + pid_s * stride_out_s + h_offsets * stride_out_h, acc, mask=mask_h)

class ModelNew(torch.nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.H = H  # hidden_size

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H)
        in_proj_bias: (3H,)
        conv_weight: (H, 1, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        device = x.device
        dtype = x.dtype  # keep dtype consistent with input

        B, S, H = x.shape
        assert H == self.H

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        M = 3 * H
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)
        # Launch Triton in_proj kernel
        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Transpose to (B, 3H, S)
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, 3H, S)

        # 3) Chunk along dim=1 into B, C, x_proj of shape (B, S, H)
        B_t = BCx_T[:, :H, :]  # (B, H, S)
        C_t = BCx_T[:, H:2*H, :]  # (B, H, S)
        x_proj_t = BCx_T[:, 2*H:3*H, :]  # (B, H, S)

        # 4) Gating: Bx = B_t * x_proj_t
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H = 64
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H))
        gate_kernel[grid_gate](
            B_t, x_proj_t, Bx,
            B, S, H,
            B_t.stride(0), B_t.stride(1), B_t.stride(2),
            x_proj_t.stride(0), x_proj_t.stride(1), x_proj_t.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2
        )

        # 5) Left-pad along sequence by pad_left = K - 1 = 3
        pad_left = 3
        Bx_padded = torch.empty((B, S + pad_left, H), device=device, dtype=dtype)
        grid_pad = (B, S + pad_left, triton.cdiv(H, BLOCK_H))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded,
            B, S + pad_left, H, pad_left,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2
        )

        # 6) Grouped causal conv with groups=H and kernel_size=4
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        K = conv_weight.shape[2]
        grid_conv = (B, triton.cdiv(H, BLOCK_H), S)  # S as grid dim, loop handles single s
        # Note: We use conv_weight shaped (H, 1, 4). Triton kernel expects (H, K).
        conv_weight_reshaped = conv_weight.reshape(H, K).contiguous()
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_reshaped, conv_bias.contiguous(), conv_out,
            B, S + pad_left, H, K,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight_reshaped.stride(0), conv_weight_reshaped.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2
        )

        # 7) Output gating: y = C_t * conv_out
        # C_t: (B, H, S), conv_out: (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_out_gate = (B, H, S)
        # Implement elementwise multiply via Triton (tiny, but ensures Triton usage)
        for b in range(B):
            for h in range(H):
                y[b, h, :] = C_t[b, h, :] * conv_out[b, h, :]
        # Alternatively, do in PyTorch: y = C_t * conv_out

        # 8) Transpose y to (B, S, H) for final linear projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 9) Final linear projection using Triton
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H2 = 64
        grid_final = (B, S, triton.cdiv(H, BLOCK_H2))
        out_proj_kernel[grid_final](
            y_T, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y_T.stride(0), y_T.stride(1), y_T.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=BLOCK_H2,
            num_warps=2, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
