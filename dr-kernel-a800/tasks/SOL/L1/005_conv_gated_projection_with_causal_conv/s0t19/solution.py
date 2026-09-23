import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,        # *f32, shape (B, S, H)
    W_ptr,        # *f32, shape (M, H) where M=3*H
    BIAS_ptr,     # *f32, shape (M,)
    OUT_ptr,      # *f32, shape (B, S, M)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,             # M = 3 * H
    BLOCK_S: tl.constexpr,       # tile along S
    BLOCK_M: tl.constexpr,       # tile along M (channels)
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(S/BLOCK_S))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    s_mask = s_offsets < S
    m_mask = m_offsets < M

    # Accumulator for each (m, s) tile
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over H
    for h in range(0, H):
        # X[b, s, h] as vector over s_offsets
        x_ptrs = X_ptr + pid_b * S * H + s_offsets * H + h
        x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)  # [BLOCK_S]

        # W[m, h] as vector over m_offsets
        w_ptrs = W_ptr + m_offsets * H + h               # W is row-major (M, H)
        w_vals = tl.load(w_ptrs, mask=m_mask, other=0.0)  # [BLOCK_M]

        # Outer product: x_vals[:, None] * w_vals[None, :]
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias: bias[m] broadcast over s
    bias_vals = tl.load(BIAS_ptr + m_offsets, mask=m_mask, other=0.0)  # [BLOCK_M]
    acc += bias_vals[None, :]

    # Store to OUT[b, s, m]
    out_ptrs = OUT_ptr + pid_b * S * M + s_offsets[:, None] * M + m_offsets[None, :]
    tl.store(out_ptrs, acc, mask=(s_mask[:, None] & m_mask[None, :]))


@triton.jit
def pad_s_left_kernel(
    IN_ptr,       # *f32, shape (B, H, S)
    OUT_ptr,      # *f32, shape (B, H, S_padded)
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    S_padded: tl.constexpr,      # S + pad_left
    pad_left: tl.constexpr,      # e.g., 3 for K=4
    MAX_S_padded: tl.constexpr,  # upper bound, we can set 8200 safely
):
    # We process each (b, h, t) and for t < pad_left, write 0; otherwise OUT[b, h, t - pad_left] = IN[b, h, t]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)  # can be up to S_padded

    t = pid_t
    # Guard against exceeding S_padded
    if t >= S_padded:
        return

    is_pad = t < pad_left
    # For non-pad positions, copy from input
    if not is_pad:
        inp_ptrs = IN_ptr + pid_b * H * S + pid_h * S + t
        out_ptrs = OUT_ptr + pid_b * H * S_padded + pid_h * S_padded + t
        val = tl.load(inp_ptrs)
        tl.store(out_ptrs, val)


@triton.jit
def conv1d_groupsB_kernel(
    IN_ptr,       # *f32, padded input (B, H, S_padded)
    W_ptr,        # *f32, weight reshaped to (H, K), K=4
    BIAS_ptr,     # *f32, (H,)
    OUT_ptr,      # *f32, (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,             # output sequence length
    S_padded: tl.constexpr,      # input sequence length (S + pad_left)
    K: tl.constexpr,             # kernel size, 4 here
    pad_left: tl.constexpr,      # 3 for K=4
    MAX_S_padded: tl.constexpr,  # upper bound
):
    # Each program handles one (b, c, t) output
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # We only process t in [0, S)
    t = pid_t
    acc = 0.0

    # Sum over K taps: weight[c, k] * IN[b, c, t + k] for k in [0..K-1], with causal (t+k < S_padded)
    for k in range(0, K):
        k_int = k  # k is constexpr, but we keep it explicit
        pos = t + k_int
        if pos < S_padded:
            inp_ptrs = IN_ptr + pid_b * H * S_padded + pid_c * S_padded + pos
            inp_val = tl.load(inp_ptrs)
            w_val = tl.load(W_ptr + pid_c * K + k_int)
            acc += inp_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_c)
    acc += bias_val

    # Store to OUT[b, c, t]
    out_ptrs = OUT_ptr + pid_b * H * S + pid_c * S + t
    tl.store(out_ptrs, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,        # *f32, (B, S, H)
    W_ptr,        # *f32, (H, H) row-major
    BIAS_ptr,     # *f32, (H,)
    OUT_ptr,      # *f32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,       # tile along H (e.g., 64 or 128)
):
    # Grid: (B, S, ceil(H/BLOCK_H))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_offsets < H

    # accumulator for the output vector over H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduction over input channels h2 (rows of W)
    for h2 in range(0, H):
        # Y[b, s, h2] scalar
        y_index = pid_b * S * H + pid_s * H + h2
        y_val = tl.load(Y_ptr + y_index)

        # W[h2, h_offsets] vector
        w_ptrs = W_ptr + h2 * H + h_offsets  # W is (H, H), row-major
        w_vec = tl.load(w_ptrs, mask=h_mask, other=0.0)

        acc += y_val * w_vec

    # Add bias
    bias_vec = tl.load(BIAS_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vec

    # Store OUT[b, s, h_offsets]
    out_base = pid_b * S * H + pid_s * H + h_offsets
    tl.store(OUT_ptr + out_base, acc, mask=h_mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H), in_proj_bias: (3H,)
        conv_weight: (H, 1, 4), conv_bias: (H,)
        out_proj_weight: (H, H), out_proj_bias: (H,)
        All tensors are expected to be float32 and contiguous.
        """
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        M = 3 * H
        K = conv_weight.shape[2]
        pad_left = K - 1  # causal left-pad

        # 1) Initial linear projection: BCx = in_proj(x, in_proj_weight, in_proj_bias)
        x_in = x.contiguous().to(dtype)
        W_in = in_proj_weight.contiguous().to(dtype)   # (M, H)
        bias_in = in_proj_bias.contiguous().to(dtype)  # (M,)

        BCx = torch.empty((B, S, M), device=device, dtype=dtype)

        BLOCK_S = 128
        BLOCK_M = 128
        grid_in = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(S, BLOCK_S))
        in_proj_linear_kernel[grid_in](
            x_in, W_in, bias_in, BCx,
            B=B, S=S, H=H, M=M,
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj along last dim (channels), each (B, S, H)
        # Triton not needed for split; torch ops for correctness:
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2*H]
        x_proj_t = BCx[:, :, 2*H:3*H]

        # 3) Element-wise gating: Bx = B_t * x_proj_t (shape: B, H, S)
        # Convert to desired order for conv: (B, H, S)
        Bx = (B_t * x_proj_t).transpose(-1, -2).contiguous()  # (B, H, S)

        # 4) Left-pad along sequence for causal conv: pad 3 elements on the left
        S_padded = S + pad_left
        Bx_padded = torch.empty((B, H, S_padded), device=device, dtype=dtype)

        grid_pad = (B, H, S_padded)
        pad_s_left_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, H=H, S=S, S_padded=S_padded, pad_left=pad_left,
            MAX_S_padded=8200,  # upper bound; Triton allows constexpr
            num_warps=2, num_stages=1
        )

        # 5) Grouped 1D conv with groups=B, kernel_size=K, bias per channel
        # conv_weight: (H, 1, 4) -> reshape to (H, K)
        conv_weight_r = conv_weight.reshape(H, K).contiguous().to(dtype)  # (H, 4)
        conv_bias_r = conv_bias.contiguous().to(dtype)  # (H,)

        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)  # output length S

        grid_conv = (B, H, S)  # third dim is S
        conv1d_groupsB_kernel[grid_conv](
            Bx_padded, conv_weight_r, conv_bias_r, conv_out,
            B=B, H=H, S=S, S_padded=S_padded, K=K, pad_left=pad_left, MAX_S_padded=8200,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out
        y = C_t * conv_out  # (B, H, S)

        # 7) Transpose back to (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 8) Final linear projection using Triton out_proj
        out_proj_w = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_proj_b = out_proj_bias.contiguous().to(dtype)    # (H,)

        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out = torch.empty((B, S, H), device=device, dtype=dtype)

        out_proj_linear_kernel[grid_out](
            y_T, out_proj_w, out_proj_b, out,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
