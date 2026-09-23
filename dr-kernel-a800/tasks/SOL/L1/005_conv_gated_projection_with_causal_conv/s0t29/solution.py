import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    x_ptr,                # *float32, input x: (B, S, H), contiguous
    in_proj_w_ptr,        # *float32, in_proj_weight: (M, H), contiguous, M = 3H
    in_proj_b_ptr,        # *float32, in_proj_bias: (M,), contiguous
    bcx_ptr,              # *float32, output BCx: (B, S, M), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Accumulator for BCx[b, s, m_offsets]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Loop over hidden dimension H
    for h in range(0, H):
        # Load x[b, s, h] as scalar
        x_val = tl.load(x_ptr + b * S * H + s * H + h, mask=True, other=0.0)  # scalar float32
        # Load in_proj_weight[m, h] vector over m_offsets
        w_ptr = in_proj_w_ptr + m_offsets * H + h  # (M,) indexed by m_offsets
        w = tl.load(w_ptr, mask=mask_m, other=0.0)  # [BLOCK_M] float32
        acc += w * x_val

    # Add bias
    b_ptr = in_proj_b_ptr + m_offsets  # (M,)
    bias = tl.load(b_ptr, mask=mask_m, other=0.0)
    acc += bias

    # Store BCx[b, s, m_offsets]
    out_ptr = bcx_ptr + b * (S * M) + s * M + m_offsets
    tl.store(out_ptr, acc, mask=mask_m)


@triton.jit
def conv_groupsH_kernel(
    bx_ptr,               # *float32, input after gating and left-pad: (B, S_padded, H), contiguous
    conv_w_ptr,           # *float32, conv_weight: (H, 1, K), contiguous, but we pass (H, K)
    conv_b_ptr,           # *float32, conv_bias: (H,), contiguous
    conv_out_ptr,         # *float32, output: (B, H, S), contiguous
    B: tl.constexpr,
    S_padded: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)           # channel index in [0..H-1]
    t = tl.program_id(2)           # time index in [0..S-1] (we output S)

    # We compute conv_out[b, c, t] = sum_{k=0..K-1} bx[b, c, t + k] * conv_w[c, k] + conv_bias[c]
    # bx_ptr is (B, S_padded, H), conv_w_ptr is (H, K). We pass conv_w as (H, K) contiguous.
    # For fixed (b, c), t in [0..S-1], t+k in [0..S_padded-1] due to pad.
    sum_val = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        pos = t + k
        # Valid if pos < S_padded
        valid = pos < S_padded
        # Load bx[b, c, pos]
        bx_ptr_pos = bx_ptr + b * (S_padded * H) + c * S_padded + pos
        bx_val = tl.load(bx_ptr_pos, mask=valid, other=0.0)
        # Load conv_w[c, k]
        w_val = tl.load(conv_w_ptr + c * K + k)  # scalar
        sum_val += bx_val * w_val

    # Add bias
    bias_val = tl.load(conv_b_ptr + c)
    sum_val += bias_val

    # Store conv_out[b, c, t]
    out_ptr = conv_out_ptr + b * (H * S) + c * S + t
    tl.store(out_ptr, sum_val)


@triton.jit
def out_proj_kernel(
    y_ptr,                # *float32, input y: (B, S, H), contiguous (we pass y_T here)
    out_w_ptr,            # *float32, out_proj_weight: (H, H), contiguous
    out_b_ptr,            # *float32, out_proj_bias: (H,), contiguous
    output_ptr,           # *float32, output: (B, S, H), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    # Each program computes one output element: output[b, s, h]
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    # Compute dot over h2 in H
    dot = tl.zeros((), dtype=tl.float32)
    for h2 in range(0, H):
        y_val = tl.load(y_ptr + b * (S * H) + s * H + h2)  # y[b, s, h2]
        w_val = tl.load(out_w_ptr + h * H + h2)           # out_w[h, h2]
        dot += y_val * w_val
    # Add bias
    bias_val = tl.load(out_b_ptr + h)
    dot += bias_val

    # Store
    tl.store(output_ptr + b * (S * H) + s * H + h, dot)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the original pipeline:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        2) Split: B = BCx[:, :, :H], C = BCx[:, :, H:2H], x_proj = BCx[:, :, 2H:3H]
        3) Gating: Bx = B * x_proj -> (B, S, H)
        4) Grouped causal conv: conv_weight (H, 1, 4), pad_left=3 -> conv_out (B, H, S)
        5) Output gating: y = C * conv_out -> (B, H, S)
        6) Final linear: y -> (B, S, H) via out_proj_weight (H, H), out_proj_bias (H,)
        """
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        device = x.device
        dtype = torch.float32
        B, S, H = x.shape
        M = 3 * H

        # Ensure float32 and contiguous
        x_c = x.contiguous().to(dtype)
        in_proj_w = in_proj_weight.contiguous().to(dtype)  # (M, H)
        in_proj_b = in_proj_bias.contiguous().to(dtype)    # (M,)
        # Allocate BCx (B, S, M)
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)

        # Launch in_proj kernel
        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_c, in_proj_w, in_proj_b, BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # Split BCx into B, C, x_proj (all (B, S, H))
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # Elementwise gating: Bx = B_t * x_proj
        # Implement in Triton
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_gate = (B, S, H)
        # Triton elementwise kernel: y[b, s, h] = B_t[b, s, h] * x_proj[b, s, h]
        @triton.jit
        def gate_kernel(y_ptr, a_ptr, b_ptr, B: tl.constexpr, S: tl.constexpr, H: tl.constexpr):
            b = tl.program_id(0)
            s = tl.program_id(1)
            h = tl.program_id(2)
            a = tl.load(a_ptr + b * (S * H) + s * H + h)
            bb = tl.load(b_ptr + b * (S * H) + s * H + h)
            y = a * bb
            tl.store(y_ptr + b * (S * H) + s * H + h, y)
        gate_kernel[grid_gate](Bx, B_t, x_proj, B=B, S=S, H=H, num_warps=1, num_stages=1)

        # Pad Bx by pad_left=K-1=3 on the left to get Bx_padded: (B, S+3, H)
        K = conv_weight.shape[2]
        pad_left = K - 1
        S_padded = S + pad_left
        Bx_padded = torch.empty((B, S_padded, H), device=device, dtype=dtype)
        # Initialize with zeros
        Bx_padded.zero_()
        if S > 0:
            # Copy Bx into Bx_padded[..., pad_left:]
            # Destination index range [pad_left .. S_padded-1]
            # Source index range [0 .. S-1], both length S
            # We can write: Bx_padded[:, pad_left:pad_left+S, :] = Bx
            Bx_padded[:, pad_left:pad_left + S, :] = Bx

        # Prepare conv_weight as (H, K) contiguous: conv_weight is (H, 1, K) -> reshape to (H, K)
        conv_w = conv_weight.reshape(H, K).contiguous().to(dtype)  # (H, K)
        conv_b = conv_bias.contiguous().to(dtype)                  # (H,)

        # Allocate conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)

        # Launch grouped conv kernel
        grid_conv = (B, H, S)
        conv_groupsH_kernel[grid_conv](
            Bx_padded, conv_w, conv_b, conv_out,
            B=B, S_padded=S_padded, H=H, K=K,
            num_warps=2, num_stages=2
        )

        # Output gating: y = C_t * conv_out -> (B, H, S)
        # Implement in Triton
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_out = (B, H, S)
        @triton.jit
        def gate2_kernel(y_ptr, a_ptr, b_ptr, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr):
            b = tl.program_id(0)
            h = tl.program_id(1)
            s = tl.program_id(2)
            a = tl.load(a_ptr + b * (H * S) + h * S + s)
            bb = tl.load(b_ptr + b * (H * S) + h * S + s)
            y = a * bb
            tl.store(y_ptr + b * (H * S) + h * S + s, y)
        gate2_kernel[grid_out](y, C_t, conv_out, B=B, H=H, S=S, num_warps=1, num_stages=1)

        # Transpose back to (B, S, H) for final linear
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # Final linear projection using Triton: y_T (B, S, H) -> output (B, S, H)
        out_w = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_b = out_proj_bias.contiguous().to(dtype)    # (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H = 64
        grid_fin = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_fin](
            y_T, out_w, out_b, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
