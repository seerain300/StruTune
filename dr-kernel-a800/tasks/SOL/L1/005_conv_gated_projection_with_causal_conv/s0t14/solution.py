import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    X_ptr,          # *f32, shape (B, S, H)
    W_ptr,          # *f32, shape (3H, H)
    Bias_ptr,       # *f32, shape (3H,)
    OUT_ptr,        # *f32, shape (B, S, 3H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,  # M = 3 * H
    BLOCK_M: tl.constexpr,  # tile size for M
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m_tile = tl.program_id(2)
    m_offsets = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # accumulator for each m in the tile
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over hidden dimension H to accumulate dot product
    for h in range(0, H):
        # load x[b, s, h]
        x_index = pid_b * S * H + pid_s * H + h  # x is contiguous with leading dimension B*S*H
        x_val = tl.load(X_ptr + x_index)
        # load W[m_offsets, h] for all m in tile
        w_index = m_offsets * H + h
        w_vec = tl.load(W_ptr + w_index, mask=m_mask, other=0.0)
        acc += x_val * w_vec

    # add bias
    bias_vec = tl.load(Bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vec

    # store to OUT[b, s, m_offsets]
    out_base = pid_b * S * M + pid_s * M + m_offsets
    tl.store(OUT_ptr + out_base, acc, mask=m_mask)


@triton.jit
def pad_left_kernel(
    IN_ptr,      # *f32, shape (B, S, H)
    OUT_ptr,     # *f32, shape (B, S_padded, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    PAD_LEFT: tl.constexpr,  # K - 1, e.g., 3
    S_padded: tl.constexpr,  # S + PAD_LEFT
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)  # iterate over padded sequence positions
    pid_h = tl.program_id(2)  # iterate over hidden channels

    h = pid_h  # scalar
    # check bounds
    if h >= H:
        return

    # If position is within padded but before original data, write 0
    is_pad = pid_t < PAD_LEFT

    if not is_pad:
        # copy from original input: t = pid_t - PAD_LEFT
        t = pid_t - PAD_LEFT
        in_index = pid_b * S * H + t * H + h
        val = tl.load(IN_ptr + in_index)
    else:
        val = 0.0

    out_index = pid_b * S_padded * H + pid_t * H + h
    tl.store(OUT_ptr + out_index, val)


@triton.jit
def conv1d_groupsH_kernel(
    IN_ptr,        # *f32, Bx_padded, shape (B, S_padded, H)
    WEIGHT_ptr,    # *f32, conv_weight, shape (H, 4) row-major (H, K)
    BIAS_ptr,      # *f32, shape (H,)
    OUT_ptr,       # *f32, conv_out, shape (B, H, S)
    B: tl.constexpr,
    S_padded: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,  # kernel_size, here 4
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # channel index in [0..H-1]
    pid_t = tl.program_id(2)  # output position in sequence [0..S-1]

    # We are computing conv_out[b, c, pid_t] = sum_{k=0..K-1} Bx_padded[b, c, pid_t + k] * weight[c, k] + bias[c]
    acc = 0.0
    # Accumulate over taps
    for k in range(0, K):
        t_actual = pid_t + k  # valid for k in [0..K-1], pid_t in [0..S-1]
        in_index = pid_b * S_padded * H + t_actual * H + pid_c
        val = tl.load(IN_ptr + in_index)
        w_index = pid_c * K + k
        w_val = tl.load(WEIGHT_ptr + w_index)
        acc += val * w_val
    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_c)
    acc += bias_val

    out_index = pid_b * H * S + pid_c * S + pid_t
    tl.store(OUT_ptr + out_index, acc)


@triton.jit
def out_proj_kernel(
    Y_ptr,         # *f32, (B, S, H)  (this is C * conv_out computed below)
    W_ptr,         # *f32, (H, H)
    Bias_ptr,      # *f32, (H,)
    OUT_ptr,       # *f32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h_tile = tl.program_id(2)

    h_offsets = pid_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduction over input channels (h2 dimension)
    for h2 in range(0, H):
        y_index = pid_b * S * H + pid_s * H + h2
        y_val = tl.load(Y_ptr + y_index)
        w_index = h2 * H + h_offsets  # W is row-major (H, H)
        w_vec = tl.load(W_ptr + w_index, mask=h_mask, other=0.0)
        acc += y_val * w_vec

    # Add bias
    bias_vec = tl.load(Bias_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vec

    # Store OUT[b, s, h_offsets]
    out_base = pid_b * S * H + pid_s * H + h_offsets
    tl.store(OUT_ptr + out_base, acc, mask=h_mask)


@triton.jit
def gate_and_mul_kernel(
    B_ptr,         # *f32, (B, S, H)
    X_ptr,         # *f32, (B, S, H)
    OUT_ptr,       # *f32, (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Compute OUT = B * X elementwise over (B, S, H).
    Grid (B, S, ceil(H / BLOCK_H))
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h_tile = tl.program_id(2)
    h_offsets = pid_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Load B and X
    b_index = pid_b * S * H + pid_s * H + h_offsets
    x_index = pid_b * S * H + pid_s * H + h_offsets
    b_vec = tl.load(B_ptr + b_index, mask=h_mask, other=0.0)
    x_vec = tl.load(X_ptr + x_index, mask=h_mask, other=0.0)
    out_vec = b_vec * x_vec

    tl.store(OUT_ptr + b_index, out_vec, mask=h_mask)


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
        All tensors are expected to be float32 and contiguous.
        """

        device = x.device
        dtype = torch.float32

        # Ensure contiguity and dtype
        x = x.contiguous().to(dtype)                     # (B, S, H)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)  # (3H, H)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)      # (3H,)
        conv_weight = conv_weight.contiguous().to(dtype)        # (H, 1, 4)
        conv_bias = conv_bias.contiguous().to(dtype)            # (H,)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)      # (H,)

        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel_size, expected 4
        S_padded = S + (K - 1)

        # 1) Initial linear projection: BCx = in_proj(x, in_proj_weight, in_proj_bias)
        # Using Triton kernel
        M = 3 * H
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)
        BLOCK_M = 64
        grid_in_proj = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in_proj](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj along last dim of size H
        # These are slices/views with metadata, not compute:
        B_slice = BCx[:, :, :H]            # (B, S, H)
        C_slice = BCx[:, :, H:2*H]        # (B, S, H)
        x_proj = BCx[:, :, 2*H:3*H]       # (B, S, H)

        # 3) Gating: Bx = B_slice * x_proj (elementwise) — implement in Triton
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        BLOCK_H = 128
        grid_gate = (B, S, triton.cdiv(H, BLOCK_H))
        gate_and_mul_kernel[grid_gate](
            B_slice, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 4) Left-pad Bx by K-1 (causal) to make S_padded
        Bx_padded = torch.empty((B, S_padded, H), device=device, dtype=dtype)
        grid_pad = (B, S_padded, H)
        pad_left = K - 1
        pad_left_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, S=S, H=H, PAD_LEFT=pad_left, S_padded=S_padded,
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal conv1d with groups=H and weight (H, 1, 4)
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_conv = (B, H, S)
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight.view(H, K), conv_bias,
            conv_out,
            B=B, S_padded=S_padded, H=H, K=K,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_slice * conv_out
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        BLOCK_H = 128
        grid_gate2 = (B, S, triton.cdiv(H, BLOCK_H))
        gate_and_mul_kernel[grid_gate2](
            C_slice, conv_out, y,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 7) Final linear projection: F.linear(y, out_proj_weight, out_proj_bias)
        # Implement via Triton kernel: output[b, s, h] = sum_{h2} y[b, h2, s] * out_proj_weight[h2, h] + out_proj_bias[h]
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_out = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
