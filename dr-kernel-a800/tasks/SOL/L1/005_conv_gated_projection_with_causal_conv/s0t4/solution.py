import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Kernel 1: Initial Linear Projection (in_proj): OUT[b, s, m] = sum_h W[m, h] * X[b, s, h] + bias[m]
# Shapes:
#   X: (B, S, H)
#   W: (M, H), M=3*H
#   bias: (M,)
#   OUT: (B, S, M)
# Launch grid: (B, ceil(S/BLOCK_S), ceil(M/BLOCK_M))
# -----------------------------
@triton.jit
def in_proj_kernel(
    X_ptr,         # *f32, (B, S, H)
    W_ptr,         # *f32, (M, H)
    Bias_ptr,      # *f32, (M,)
    OUT_ptr,       # *f32, (B, S, M)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,             # = 3*H
    BLOCK_S: tl.constexpr,       # e.g., 128
    BLOCK_M: tl.constexpr,       # e.g., 64
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Compute offsets for this tile
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    # Masks for bounds
    s_mask = s_offsets < S
    m_mask = m_offsets < M

    # Initialize accumulator: (BLOCK_S, BLOCK_M)
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over H
    for h in range(0, H):
        # Load x for this h across s tile
        x_base = pid_b * S * H + s_offsets * H + h  # shape (BLOCK_S,)
        x_vec = tl.load(X_ptr + x_base, mask=s_mask, other=0.0)  # (BLOCK_S,)

        # Load W for this h across m tile
        # W layout is (M, H) row-major
        w_base = m_offsets * H + h  # shape (BLOCK_M,)
        w_vec = tl.load(W_ptr + w_base, mask=m_mask, other=0.0)  # (BLOCK_M,)

        # Outer product: (BLOCK_S, 1) * (1, BLOCK_M)
        acc += x_vec[:, None] * w_vec[None, :]

    # Add bias: bias shape (M,)
    bias_vec = tl.load(Bias_ptr + m_offsets, mask=m_mask, other=0.0)  # (BLOCK_M,)
    acc += bias_vec[None, :]  # broadcast over s

    # Store OUT[b, s, m] at this tile
    out_base = pid_b * S * M + s_offsets[:, None] * M + m_offsets[None, :]  # (BLOCK_S, BLOCK_M)
    store_mask = s_mask[:, None] & m_mask[None, :]
    tl.store(OUT_ptr + out_base, acc, mask=store_mask)


# -----------------------------
# Kernel 2: Left-pad along sequence dimension: OUT[b, c, t] = IN[b, c, t - pad_left] if t >= pad_left else 0
# Inputs:
#   IN: (B, C, S)
#   pad_left: int
# Outputs:
#   OUT: (B, C, S_out) with S_out = S + pad_left
# Launch grid: (B, C, S_out)
# -----------------------------
@triton.jit
def left_pad_kernel(
    IN_ptr,    # *f32, (B, C, S)
    OUT_ptr,   # *f32, (B, C, S_out)
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    pad_left: tl.constexpr,
    S_out: tl.constexpr,  # = S + pad_left
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    t = tl.program_id(2)

    # Compute input index: if t < pad_left, value is 0; else shift by pad_left
    if t < pad_left:
        out_val = 0.0
    else:
        in_index = pid_b * C * S + pid_c * S + (t - pad_left)
        out_val = tl.load(IN_ptr + in_index)

    out_index = pid_b * C * S_out + pid_c * S_out + t
    tl.store(OUT_ptr + out_index, out_val)


# -----------------------------
# Kernel 3: Grouped 1D Convolution (groups = H), K=4, no padding, stride=1
# We assume input is already left-padded on host to S_padded = S + (K-1).
# Inputs:
#   IN: (B, H, S_padded)  <- Bx_padded
#   Weight: (H, K) where Weight[c, k] = conv_weight[c, 0, k]
#   Bias: (H,)
# Outputs:
#   OUT: (B, H, S)  conv_out
# Launch grid: (B, H, S)  (one program per (b, c, t))
# -----------------------------
@triton.jit
def conv1d_grouped_kernel(
    IN_ptr,     # *f32, (B, H, S_padded)
    WEIGHT_ptr, # *f32, (H, K), row-major
    BIAS_ptr,   # *f32, (H,)
    OUT_ptr,    # *f32, (B, H, S)
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,            # output length
    S_padded: tl.constexpr,     # input length after padding
    K: tl.constexpr,            # kernel size, here 4
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    t = tl.program_id(2)  # output time index

    # Accumulator for this (b, c, t)
    acc = 0.0

    # Sum over K taps
    for k in range(0, K):
        # IN[b, c, t + k] is valid because host passes t in [0, S) and S_padded = S + K - 1
        in_index = pid_b * H * S_padded + pid_c * S_padded + (t + k)
        weight_index = pid_c * K + k
        w = tl.load(WEIGHT_ptr + weight_index)
        val = tl.load(IN_ptr + in_index)
        acc += val * w

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_c)
    acc += bias_val

    # Store
    out_index = pid_b * H * S + pid_c * S + t
    tl.store(OUT_ptr + out_index, acc)


# -----------------------------
# Kernel 4: Final Linear (out_proj): OUT[b, s, h] = sum_h2 Y[b, s, h2] * W[h2, h] + Bias[h]
# Inputs:
#   Y: (B, S, H)
#   W: (H, H)
#   Bias: (H,)
# Outputs:
#   OUT: (B, S, H)
# Launch grid: (B, S, ceil(H/BLOCK_H))
# Each program handles one (b, s) and a tile of h dimension.
# -----------------------------
@triton.jit
def out_proj_kernel(
    Y_ptr,      # *f32, (B, S, H)
    W_ptr,      # *f32, (H, H)
    Bias_ptr,   # *f32, (H,)
    OUT_ptr,    # *f32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,      # tile size along H (e.g., 64 or 128)
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h_tile = tl.program_id(2)

    h_offsets = pid_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_offsets < H

    # Accumulator for h tile
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduction over h2 (input channel dimension)
    for h2 in range(0, H):
        # Load Y[b, s, h2]
        y_index = pid_b * S * H + pid_s * H + h2
        y_val = tl.load(Y_ptr + y_index)

        # Load W[h2, h_offsets]
        w_index = h2 * H + h_offsets  # row-major W: (H, H)
        w_vec = tl.load(W_ptr + w_index, mask=h_mask, other=0.0)

        # Fused multiply-add
        acc += y_val * w_vec

    # Add bias
    bias_vec = tl.load(Bias_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vec

    # Store OUT[b, s, h_offsets]
    out_base = pid_b * S * H + pid_s * H + h_offsets
    tl.store(OUT_ptr + out_base, acc, mask=h_mask)


# -----------------------------
# ModelNew: orchestrator that uses Triton kernels
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

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
        # Enforce dtype and contiguity
        device = x.device
        B, S, H = x.shape
        K = conv_weight.shape[2]
        pad_left = K - 1
        S_padded = S + pad_left

        # 1) Initial linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # We implement it in Triton.
        M = 3 * H
        x_in = x.contiguous().to(torch.float32)
        W_in = in_proj_weight.contiguous().to(torch.float32)   # (M, H)
        bias_in = in_proj_bias.contiguous().to(torch.float32)  # (M,)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)

        # Launch in_proj_kernel with a 3D grid
        BLOCK_S = 128
        BLOCK_M = 64
        grid_in = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_in, W_in, bias_in, BCx,
            B=B, S=S, H=H, M=M,
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj along last dim of size H
        #   This is a view-like split. We do not need to compute it via Triton since it's metadata.
        B_group = BCx[:, :, :H]
        C_group = BCx[:, :, H:2 * H]
        x_proj_group = BCx[:, :, 2 * H:3 * H]

        # 3) Element-wise gating: Bx = B_group * x_proj_group
        #    Shapes: (B, H, S). Note: In original code, Bx = B * x_proj with B and x_proj being (B,H,S).
        #    Here, we have (B,S,H) groups, but after splitting along last dim, each is (B,H,S).
        #    The original code splits BCx along channel dimension (size H), not along S.
        #    We need to confirm split semantics: BCx has shape (B, S, 3H); splitting along last dim (channels) yields 3 chunks each of size H, but the code uses:
        #      B = BCx[:, :, :H], C = BCx[:, :, H:2H], x_proj = BCx[:, :, 2H:3H]
        #      Then Bx = B * x_proj, which implies Bx shape (B, H, S). We will follow this exactly.
        #    Elementwise multiply across dims (B,H,S): We can do this with torch to ensure correctness.
        Bx = B_group * x_proj_group  # (B, H, S)

        # 4) Transpose for conv: (B, H, S) -> (B, H, S)
        #    conv1d expects (N, C_in, L). We will pad along L before conv.

        # 5) Pad input along sequence for causal conv
        Bx_pad = torch.empty((B, H, S_padded), device=device, dtype=torch.float32)
        grid_pad = (B, H, S_padded)
        left_pad_kernel[grid_pad](
            Bx, Bx_pad,
            B=B, C=H, S=S, pad_left=pad_left, S_out=S_padded,
            num_warps=1, num_stages=1
        )

        # 6) Grouped conv1d (groups=H): conv_out (B, H, S)
        conv_weight_flat = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4) -> view (H, 4)
        conv_bias_flat = conv_bias.contiguous().to(torch.float32)      # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        grid_conv = (B, H, S)
        conv1d_grouped_kernel[grid_conv](
            Bx_pad, conv_weight_flat, conv_bias_flat, conv_out,
            B=B, H=H, S=S, S_padded=S_padded, K=K,
            num_warps=4, num_stages=2
        )

        # 7) Output gating: y = C_group * conv_out
        #    Shapes: C_group (B, H, S), conv_out (B, H, S) -> y (B, H, S)
        y = C_group * conv_out

        # 8) Transpose back to (B, S, H) for final linear
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 9) Final linear projection
        out_proj_weight_t = out_proj_weight.contiguous().to(torch.float32)   # (H, H)
        out_proj_bias_t = out_proj_bias.contiguous().to(torch.float32)       # (H,)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_H = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight_t, out_proj_bias_t, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
