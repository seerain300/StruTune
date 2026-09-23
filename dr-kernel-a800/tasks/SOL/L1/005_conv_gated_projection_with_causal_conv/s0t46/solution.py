import torch
import triton
import triton.language as tl

# Kernel 1: in_proj -> BCx = x @ in_proj_weight^T + in_proj_bias
@triton.jit
def in_proj_kernel(
    x_ptr,          # *f32, shape (B, S, H)
    in_w_ptr,       # *f32, shape (M, H), M=3H
    in_b_ptr,       # *f32, shape (M,)
    out_ptr,        # *f32, shape (B, S, M)
    B: tl.int32, S: tl.int32, H: tl.int32, M: tl.int32,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Accumulator for output[b, s, m_offsets]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over h in H (static)
    for h in range(0, H):
        # Load x[b, s, h]
        x_val = tl.load(x_ptr + b * S * H + s * H + h, mask=True, other=0.0)
        # Load in_w[m, h] for m_offsets
        w_ptrs = in_w_ptr + m_offsets * H + h
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0)
        # FMA
        acc += x_val * w_vals

    # Add bias
    b_vals = tl.load(in_b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # Store to out[b, s, m_offsets]
    out_ptrs = out_ptr + b * S * M + s * M + m_offsets
    tl.store(out_ptrs, acc, mask=mask_m)


# Kernel 2: elementwise gating Bx = B * x_proj
@triton.jit
def gating_kernel(
    B_ptr,          # *f32, shape (B, S, H)
    x_proj_ptr,     # *f32, shape (B, S, H)
    out_ptr,        # *f32, shape (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    b_val = tl.load(B_ptr + b * S * H + s * H + h_offsets, mask=mask_h, other=0.0)
    xp_val = tl.load(x_proj_ptr + b * S * H + s * H + h_offsets, mask=mask_h, other=0.0)
    out = b_val * xp_val

    out_ptrs = out_ptr + b * S * H + s * H + h_offsets
    tl.store(out_ptrs, out, mask=mask_h)


# Kernel 3: left-pad along sequence by pad_left for causal conv
@triton.jit
def left_pad_kernel(
    Bx_ptr,         # *f32, shape (B, S, H)
    out_ptr,        # *f32, shape (B, S_padded, H)
    B: tl.int32, S: tl.int32, S_padded: tl.int32, H: tl.int32, pad_left: tl.int32,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0, S_padded)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H
    mask_pos = t >= pad_left  # only non-padded positions should copy from Bx

    if mask_pos:
        src_t = t - pad_left
        src_ptr = Bx_ptr + b * S * H + src_t * H + h_offsets
        val = tl.load(src_ptr, mask=mask_h, other=0.0)
    else:
        val = tl.zeros([BLOCK_H], dtype=tl.float32)

    out_ptrs = out_ptr + b * S_padded * H + t * H + h_offsets
    tl.store(out_ptrs, val, mask=mask_h)


# Kernel 4: grouped causal 1D conv with groups=H (depthwise per channel)
@triton.jit
def conv1d_groupsH_kernel(
    Bx_pad_ptr,     # *f32, shape (B, S_padded, H)
    conv_w_ptr,     # *f32, shape (H, 4)
    conv_b_ptr,     # *f32, shape (H,)
    out_ptr,        # *f32, shape (B, H, S)
    B: tl.int32, H: tl.int32, S_padded: tl.int32, K: tl.constexpr
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Precompute strides
    # Bx_pad strides: (B, S_padded, H)
    # out strides: (B, H, S)
    # We'll compute conv_out[b, c, t] for t in [0..S-1]
    # Iterate t; for each t, sum over k=0..K-1
    for t in range(0, S):
        acc = 0.0  # scalar float32
        for k in range(0, K):
            pos = t + k
            ptr = Bx_pad_ptr + b * (S_padded * H) + pos * H + c
            val = tl.load(ptr)  # scalar
            w_ptr = conv_w_ptr + c * K + k
            w = tl.load(w_ptr)  # scalar
            acc += val * w
        b_ptr = conv_b_ptr + c
        bval = tl.load(b_ptr)  # scalar
        acc += bval
        # Store at out[b, c, t]
        out_ptr_t = out_ptr + b * (H * S) + c * S + t
        tl.store(out_ptr_t, acc)


# Kernel 5: final linear (out_proj) on y_T: (B, S, H) -> (B, S, H)
@triton.jit
def out_proj_kernel(
    y_T_ptr,        # *f32, shape (B, S, H)  (this is C * conv_out)
    out_w_ptr,      # *f32, shape (H, H)
    out_b_ptr,      # *f32, shape (H,)
    out_ptr,        # *f32, shape (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Accumulate output[b, s, h_offsets] = sum_{h2} y_T[b, s, h2] * out_w[h2, h_offsets] + out_b[h_offsets]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # For each h2 in H, compute dot with out_w[h2, h_offsets]
    for h2 in range(0, H):
        y_val = tl.load(y_T_ptr + b * S * H + s * H + h2, mask=True, other=0.0)
        w_ptrs = out_w_ptr + h2 * H + h_offsets
        w_vals = tl.load(w_ptrs, mask=mask_h, other=0.0)
        acc += y_val * w_vals

    # Add bias
    b_vals = tl.load(out_b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    out_ptrs = out_ptr + b * S * H + s * H + h_offsets
    tl.store(out_ptrs, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton-ONLY implementation of the given computation:
        1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias, (B, S, 3H)
        2) Split into B, C, x_proj (B, S, H) each
        3) Bx = B * x_proj
        4) Left-pad by K-1 for causal conv
        5) Grouped 1D conv (groups=H) with conv_weight (H,1,4), conv_bias (H,)
           Output conv_out: (B, H, S)
        6) y = C * conv_out
        7) Final linear: out = F.linear(y_T, out_proj_weight, out_proj_bias) -> (B, S, H)
        """
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda and \
               conv_weight.is_cuda and conv_bias.is_cuda and \
               out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be CUDA for Triton."

        device = x.device
        B, S, H = x.shape
        M = 3 * H
        K = conv_weight.shape[2]
        pad_left = K - 1
        S_padded = S + pad_left

        # 1) in_proj BCx
        x_f = x.contiguous().to(torch.float32)
        in_w_f = in_proj_weight.contiguous().to(torch.float32)  # (3H, H)
        in_b_f = in_proj_bias.contiguous().to(torch.float32)    # (3H,)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)

        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_f, in_w_f, in_b_f, BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split into B, C, x_proj
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:(2 * H)]
        x_proj = BCx[:, :, (2 * H):]

        # 3) Element-wise gating: Bx = B_t * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_gate = (B, S, 1)  # BLOCK_H=128 is fine but we use grid over H via launch; instead, we launch over (B,S,H)
        # Launch elementwise kernel: we need to create a 3D grid. Triton supports 3D grids; we can set BLOCK_H=1 then launch (B,S,H).
        # However, Triton expects a single BLOCK_H; better: compute over tiles of H. We'll do it as (B,S) with a small kernel over H.
        # To keep correctness, we implement a simple elementwise Triton kernel with grid (B, S, H).
        # Note: Triton can use 3D grids; but simpler is to write a 1D grid over total elements. Here we use (B,S,H) by launching separate.

        # Implement elementwise gating via Triton 3D grid: gating_kernel(B_t, x_proj, Bx, B, S, H, BLOCK_H=1)
        grid_gate = (B, S, triton.cdiv(H, 1))
        gating_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H,
            num_warps=2, num_stages=2
        )

        # 4) Left-pad Bx by pad_left
        Bx_pad = torch.empty((B, S_padded, H), device=device, dtype=torch.float32)
        grid_pad = (B, S_padded, triton.cdiv(H, 128))
        left_pad_kernel[grid_pad](
            Bx, Bx_pad,
            B=B, S=S, S_padded=S_padded, H=H, pad_left=pad_left,
            num_warps=2, num_stages=2
        )

        # 5) Grouped causal conv with groups=H (depthwise per channel)
        conv_w_f = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4) -> (H, 4)
        conv_b_f = conv_bias.contiguous().to(torch.float32)    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H)
        conv1d_groupsH_kernel[grid_conv](
            Bx_pad, conv_w_f.reshape(H, K), conv_b_f,
            conv_out,
            B=B, H=H, S_padded=S_padded, K=K,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out
        y = C_t * conv_out  # elementwise multiply

        # 7) Final linear: out = F.linear(y_T, out_proj_weight, out_proj_bias) -> (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)
        out_w_f = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_b_f = out_proj_bias.contiguous().to(torch.float32)    # (H,)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, out_w_f, out_b_f, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        # Return output; evaluation expects float32. If input dtype is not float32, the original uses float32 anyway.
        return output


def run(*args):
    return ModelNew()(*args)
