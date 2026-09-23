import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------
# Triton kernel: grouped 1D convolution (depthwise) with causal left pad
# Input: Bx_padded (B, C, S_padded), Weight: (C, 4) where row-major (C, 4) and each row is (w0, w1, w2, w3)
# Bias: (C,) conv_bias
# Output: Conv_out (B, C, S)
# conv_out[b, c, t] = sum_{k=0..3} weight[c, k] * Bx_padded[b, c, t + k] + bias[c]
# We launch one program per (b, c) pair and iterate over output t positions. K is constexpr (4).
# -----------------------------
@triton.jit
def conv1d_depthwise_kernel(
    INPUT_ptr,    # *f32, shape (B, C, S_padded)
    WEIGHT_ptr,   # *f32, shape (C, 4), row-major: weight[c, k] at index c*4 + k
    BIAS_ptr,     # *f32, shape (C,)
    OUTPUT_ptr,   # *f32, shape (B, C, S)
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,                # output sequence length (same as input S)
    S_padded: tl.constexpr,         # S + pad_left = S + (K - 1)
    K: tl.constexpr,                # kernel_size, here 4
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # We process all output t positions with a simple loop (K is small and constexpr)
    for t in range(0, S):
        acc = 0.0
        # Sum over K taps
        for k in range(0, K):
            # inp_index = b*C*S_padded + c*S_padded + (t + k)
            inp_index = pid_b * C * S_padded + pid_c * S_padded + (t + k)
            # weight index: c*4 + k
            w_index = pid_c * K + k
            w_val = tl.load(WEIGHT_ptr + w_index)
            acc += tl.load(INPUT_ptr + inp_index) * w_val
        # Add bias
        bias_val = tl.load(BIAS_ptr + pid_c)
        acc += bias_val

        # Store to output at position t
        out_index = pid_b * C * S + pid_c * S + t
        tl.store(OUTPUT_ptr + out_index, acc)


# -----------------------------
# Triton kernel: final linear projection (out_proj)
# Y: (B, S, H) = (B, S, hidden_size), W: (H, H), Bias: (H,)
# OUT[b, s, h] = sum_{h2=0..H-1} Y[b, s, h2] * W[h2, h] + Bias[h]
# Launch grid (B, S, tiles over H). H is runtime; Triton handles loops. We set BLOCK_H for tiling.
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

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduction over h2 (input channel dimension)
    for h2 in range(0, H):
        # Y index: y[b, s, h2]
        y_index = pid_b * S * H + pid_s * H + h2
        y_val = tl.load(Y_ptr + y_index)

        # W row: W[h2, h_offsets] where W is (H, H) row-major
        w_index = h2 * H + h_offsets  # vector across h_offsets
        w_vec = tl.load(W_ptr + w_index, mask=h_mask, other=0.0)

        acc += y_val * w_vec

    # Add bias
    bias_vec = tl.load(Bias_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vec

    # Store OUT[b, s, h_offsets]
    out_base = pid_b * S * H + pid_s * H + h_offsets
    tl.store(OUT_ptr + out_base, acc, mask=h_mask)


# -----------------------------
# ModelNew: orchestrator that launches Triton kernels
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
        """
        # Ensure float32 and contiguous
        device = x.device
        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel size, asserted as 4 in original

        dtype = torch.float32
        x_in = x.contiguous().to(dtype)  # (B, S, H)

        # 1) Initial linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # BCx: (B, S, 3H)
        BCx = F.linear(x_in, in_proj_weight.contiguous().to(dtype), in_proj_bias.contiguous().to(dtype))

        # 2) Split BCx into B, C, x_proj each of shape (B, H, S)
        BCx_T = BCx.transpose(-1, -2)  # (B, 3H, S)
        B_ = BCx_T[:, :H, :]           # (B, H, S)
        C_ = BCx_T[:, H:2*H, :]        # (B, H, S)
        x_proj = BCx_T[:, 2*H:, :]     # (B, H, S)

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = B_ * x_proj  # (B, H, S)

        # 4) Causal left-pad along sequence dimension for conv (pad_left = K - 1)
        pad_left = K - 1
        Bx_padded = F.pad(Bx, (pad_left, 0))  # (B, H, S + pad_left)
        S_padded = S + pad_left

        # 5) Grouped causal 1D convolution: groups=H (depthwise per (b, c)), weight (H, 1, 4), bias (H,)
        # Output conv_out: (B, H, S) because we only keep positions 0..S-1
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)

        # Prepare weight as (H, 4) row-major for Triton
        conv_weight_4 = conv_weight.contiguous().view(H, K)  # (H, 4)
        conv_bias_ = conv_bias.contiguous().to(dtype)

        # Launch conv kernel: grid over (B, H)
        grid = (B, H)
        conv1d_depthwise_kernel[grid](
            Bx_padded, conv_weight_4, conv_bias_, conv_out,
            B=B, C=H, S=S, S_padded=S_padded, K=K,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C * conv_out
        y = C_ * conv_out  # (B, H, S)

        # 7) Final linear projection: y -> (B, S, H) via out_proj
        # y has shape (B, H, S); need to linearize to (B, S, H). We reshape accordingly.
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)
        output = F.linear(y_T, out_proj_weight.contiguous().to(dtype), out_proj_bias.contiguous().to(dtype))

        return output


def run(*args):
    return ModelNew()(*args)
