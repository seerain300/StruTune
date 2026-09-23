import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------
# Triton kernel: left-pad along the sequence dimension (B, S, H) -> (B, S_padded, H)
# Bx_padded[b, t, c] = Bx[b, t - (K - 1), c] if t >= (K - 1) else 0
# -----------------------------
@triton.jit
def pad_left_sequence_kernel(
    IN_ptr,        # *f32, input tensor (B, S, H), contiguous
    OUT_ptr,       # *f32, output tensor (B, S_padded, H), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,             # kernel_size, here 4
    S_padded: tl.constexpr,      # S + (K - 1)
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Each program writes one element (b, s, c)
    in_index = pid_b * S * H + pid_s * H + pid_c
    out_index = pid_b * S_padded * H + pid_s * H + pid_c

    # For s < K - 1, value is 0; else copy from input at position s - (K - 1)
    if pid_s < (K - 1):
        tl.store(OUT_ptr + out_index, 0.0)
    else:
        src_index = pid_b * S * H + (pid_s - (K - 1)) * H + pid_c
        val = tl.load(IN_ptr + src_index)
        tl.store(OUT_ptr + out_index, val)


# -----------------------------
# Triton kernel: grouped 1D conv with groups = H (depthwise, per (b, c))
# Input: Bx_padded (B, S_padded, H)
# Weight: (H, 4) contiguous row-major
# Bias: (H,)
# Output: Conv_out (B, S, H)
# conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, t, c] * weight[c, k] + bias[c]
# We launch grid (B, H, S) and compute per (b, c, t).
# -----------------------------
@triton.jit
def conv1d_depthwise_groupsH_kernel_3d(
    INPUT_ptr,    # *f32, (B, S_padded, H)
    WEIGHT_ptr,   # *f32, (H, 4)
    BIAS_ptr,     # *f32, (H,)
    OUTPUT_ptr,   # *f32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,               # output length (same as original seq_len)
    H: tl.constexpr,
    S_padded: tl.constexpr,        # S + (K - 1)
    K: tl.constexpr,               # kernel_size, here 4
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Accumulate over K taps
    acc = 0.0
    for k in tl.static_range(0, K):
        in_index = pid_b * S_padded * H + pid_t * H + pid_c
        w_index = pid_c * K + k  # WEIGHT is (H, 4) row-major
        w_val = tl.load(WEIGHT_ptr + w_index)
        in_val = tl.load(INPUT_ptr + in_index)
        acc += in_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_c)
    acc += bias_val

    # Store to output at position t
    out_index = pid_b * S * H + pid_t * H + pid_c
    tl.store(OUTPUT_ptr + out_index, acc)


# -----------------------------
# Triton kernel: final linear projection (out_proj) using reduction over H
# Input: Y (B, S, H), Weight (H, H), Bias (H)
# Output: OUT (B, S, H)
# OUT[b, s, h] = sum_{h2 in H} Y[b, s, h2] * Weight[h, h2] + Bias[h]
# We use 3D grid (B, S, tiles of H) and static reduction over H.
# -----------------------------
@triton.jit
def out_proj_kernel(
    Y_ptr,      # *f32, (B, S, H)
    W_ptr,      # *f32, (H, H), row-major: W[h, h2] = W_ptr[h * H + h2]
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
    for h2 in tl.static_range(0, H):
        y_index = pid_b * S * H + pid_s * H + h2
        y_val = tl.load(Y_ptr + y_index)
        # Load W[h, h2] for all BLOCK_H offsets
        w_index = h2 * H + h_offsets  # W is row-major (H, H)
        w_vec = tl.load(W_ptr + w_index, mask=h_mask, other=0.0)
        acc += y_val * w_vec

    # Add bias
    bias_vec = tl.load(Bias_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vec

    # Store OUT[b, s, h_offsets]
    out_base = pid_b * S * H + pid_s * H + h_offsets
    tl.store(OUT_ptr + out_base, acc, mask=h_mask)


# -----------------------------
# ModelNew: orchestrator using Triton for heavy ops
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

        device = x.device
        dtype = torch.float32

        # Ensure contiguity and dtype
        x = x.contiguous().to(dtype)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)
        conv_weight = conv_weight.contiguous().to(dtype)  # (H, 1, 4)
        conv_bias = conv_bias.contiguous().to(dtype)      # (H,)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)      # (H,)

        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel_size, expected 4
        S_padded = S + (K - 1)

        # Step A: Initial linear projection using torch (simple and correct)
        # BCx = F.linear(x, in_proj_weight, in_proj_bias) where in_proj_weight: (3H, H)
        M = 3 * H
        BCx = F.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Split BCx into B, C, x_proj along last dim of size H
        B_slice = BCx[:, :, :H]            # (B, S, H)
        C_slice = BCx[:, :, H:2*H]        # (B, S, H)
        x_proj = BCx[:, :, 2*H:3*H]       # (B, S, H)

        # Step B: Element-wise gating: Bx = B_slice * x_proj
        # Note: This is elementwise multiply between (B, S, H) and (B, S, H).
        Bx = B_slice * x_proj  # (B, S, H)

        # Step C: Left-pad Bx along sequence dim by K-1 to make S_padded
        Bx_padded = torch.empty((B, S_padded, H), device=device, dtype=dtype)
        grid_pad = (B, S_padded, H)
        pad_left_sequence_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, S=S, H=H, K=K, S_padded=S_padded,
            num_warps=4, num_stages=2
        )

        # Step D: Grouped causal 1D conv with groups


def run(*args):
    return ModelNew()(*args)
