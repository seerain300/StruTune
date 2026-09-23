import torch
import triton
import triton.language as tl


# -----------------------------
# Kernel 1: Linear projection (in_proj) using matmul-style
# Y[b, s, m] = sum_n X[b, s, n] * W[m, n] + bias[m]
# X: (B, S, N), W: (M, N), Y: (B, S, M)
# All sizes passed as constexpr to allow static loops.
# -----------------------------
@triton.jit
def in_proj_matmul_kernel(
    X_ptr,        # *f32, shape (B, S, N)
    W_ptr,        # *f32, shape (M, N)
    BIAS_ptr,     # *f32, shape (M,)
    Y_ptr,        # *f32, shape (B, S, M)
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # seq_len
    N: tl.constexpr,   # hidden_size (input dim for in_proj)
    M: tl.constexpr,   # 3 * hidden_size (output dim for in_proj)
    BLOCK_S: tl.constexpr,  # tile size along S
    BLOCK_M: tl.constexpr,  # tile size along M
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    s_mask = s_offsets < S
    m_mask = m_offsets < M

    # Accumulator (BLOCK_S x BLOCK_M)
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Static reduction over N
    for n in tl.static_range(0, N):
        # X[b, s, n] -> shape (BLOCK_S,)
        x_ptrs = X_ptr + pid_b * S * N + s_offsets * N + n
        x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)

        # W[m, n] -> shape (BLOCK_M,)
        w_ptrs = W_ptr + m_offsets * N + n  # W is row-major: (M, N)
        w_vals = tl.load(w_ptrs, mask=m_mask, other=0.0)  # (BLOCK_M,)

        # Outer product accumulate
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_vals = tl.load(BIAS_ptr + m_offsets, mask=m_mask, other=0.0)  # (BLOCK_M,)
    acc = acc + bias_vals[None, :]

    # Store to Y[b, s, m]
    y_ptrs = Y_ptr + pid_b * S * M + s_offsets[:, None] * M + m_offsets[None, :]
    store_mask = (s_mask[:, None]) & (m_mask[None, :])
    tl.store(y_ptrs, acc, mask=store_mask)


# -----------------------------
# Kernel 2: Left-pad along the last dimension (sequence)
# Input Bx: (B, C, S_in), Output: (B, C, S_in + pad_left)
# For each (b, c, t_out):
#   if t_out < pad_left: out[b, c, t_out] = 0
#   else: out[b, c, t_out] = Bx[b, c, t_out - pad_left]
# -----------------------------
@triton.jit
def pad_left_kernel(
    IN_ptr,       # *f32, shape (B, C, S_in)
    OUT_ptr,      # *f32, shape (B, C, S_out)
    B: tl.constexpr,
    C: tl.constexpr,
    S_in: tl.constexpr,
    S_out: tl.constexpr,  # S_in + pad_left
    pad_left: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    t_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    t_mask = t_offsets < S_out

    # Determine which positions are padding
    is_pad = t_offsets < pad_left

    # For non-pad positions, map to input indices
    t_actual = t_offsets - pad_left  # valid when not pad

    # IN layout: ((b * C + c) * S_in + t)
    in_ptrs = IN_ptr + pid_b * C * S_in + pid_c * S_in + t_actual
    # OUT layout: ((b * C + c) * S_out + t)
    out_ptrs = OUT_ptr + pid_b * C * S_out + pid_c * S_out + t_offsets

    # Load input values for non-pad positions; others are zero
    in_vals = tl.load(in_ptrs, mask=(~is_pad) & t_mask, other=0.0)
    out_vals = tl.where(is_pad & t_mask, 0.0, in_vals)

    tl.store(out_ptrs, out_vals, mask=t_mask)


# -----------------------------
# Kernel 3: Grouped 1D convolution (depthwise) with groups=C
# Input: Bx_padded (B, C, S_padded), Weight: (C, 1, K), Bias: (C,)
# Output: Conv_out (B, C, S) where S is desired output length.
# We assume host has already applied causal left-padding to S_padded.
# conv_out[b, c, t] = sum_{k=0..K-1} weight[c, 0, k] * Bx_padded[b, c, t + k] + bias[c]
# Here K=4. We use static loop over t and k.
# -----------------------------
@triton.jit
def conv1d_depthwise_kernel(
    INPUT_ptr,    # *f32, shape (B, C, S_padded)
    WEIGHT_ptr,   # *f32, shape (C, K) where K=4, row-major
    BIAS_ptr,     # *f32, shape (C,)
    OUTPUT_ptr,   # *f32, shape (B, C, S)
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,               # output seq_len
    S_padded: tl.constexpr,        # S + pad_left
    K: tl.constexpr,               # kernel_size (here 4)
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Iterate over all output positions t in [0, S)
    for t in tl.static_range(0, S):
        acc = 0.0
        # Sum over K taps: input index is (b, c, t + k)
        for k in tl.static_range(0, K):
            inp_index = pid_b * C * S_padded + pid_c * S_padded + (t + k)
            w_index = pid_c * K + k
            acc += tl.load(INPUT_ptr + inp_index) * tl.load(WEIGHT_ptr + w_index)
        # Add bias
        bias_val = tl.load(BIAS_ptr + pid_c)
        acc += bias_val

        # Store to output at position t
        out_index = pid_b * C * S + pid_c * S + t
        tl.store(OUTPUT_ptr + out_index, acc)


# -----------------------------
# Kernel 4: Final linear projection (out_proj) using matmul-style
# Computes: OUT[b, s, h] = sum_h2 Y_T[b, s, h2] * W_out[h, h2] + bias_out[h]
# Y_T: (B, S, H), W_out: (H, H), OUT: (B, S, H)
# -----------------------------
@triton.jit
def out_proj_matmul_kernel(
    Y_ptr,        # *f32, shape (B, S, H)
    W_out_ptr,    # *f32, shape (H, H)
    BIAS_ptr,     # *f32, shape (H,)
    OUT_ptr,      # *f32, shape (B, S, H)
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # seq_len
    H: tl.constexpr,   # hidden_size for output projection
    BLOCK_H: tl.constexpr,  # tile along H (e.g., 128)
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduction over H2 == H
    for h2 in tl.static_range(0, H):
        # Load Y[b, s, h2] -> scalar
        y_val = tl.load(Y_ptr + pid_b * S * H + pid_s * H + h2)
        # Load W_out[h, h2] for a tile of h
        w_ptrs = W_out_ptr + h_offsets * H + h2  # (BLOCK_H,)
        w_vals = tl.load(w_ptrs, mask=h_mask, other=0.0)
        acc += y_val * w_vals

    # Add bias
    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=h_mask, other=0.0)
    acc = acc + bias_vals

    # Store OUT[b, s, h]
    out_ptrs = OUT_ptr + pid_b * S * H + h_offsets
    tl.store(out_ptrs, acc, mask=h_mask)


# -----------------------------
# ModelNew: orchestrator that launches Triton kernels
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H), in_proj_bias: (3H,)
        conv_weight: (H, 1, K), conv_bias: (H,)
        out_proj_weight: (H, H), out_proj_bias: (H,)
        All tensors are expected to be float32 and contiguous.
        """

        # Ensure float32 and contiguous for Triton kernels
        dtype = torch.float32
        B, S, H = x.shape

        # 1) Initial linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # We implement F.linear in Triton
        M = 3 * H  # 3 * hidden_size
        x_in = x.contiguous().to(dtype)                    # (B, S, H)
        W_in = in_proj_weight.contiguous().to(dtype)      # (M, H)
        bias_in = in_proj_bias.contiguous().to(dtype)     # (M,)
        BCx = torch.empty((B, S, M), device=x.device, dtype=dtype)

        # Launch in_proj_matmul_kernel: grid (B, ceil(M/BLOCK_M), ceil(S/BLOCK_S))
        BLOCK_S = 64
        BLOCK_M = 128
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(S, BLOCK_S))
        in_proj_matmul_kernel[grid](
            x_in, W_in, bias_in, BCx,
            B=B, S=S, N=H, M=M,
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # Now BCx has shape (B, S, 3H). We need B, C, x_proj. Original code chunks last dim into 3 groups.
        # Convert to (B, H, S) for each group via indexing. We'll compute B, C, x_proj in PyTorch (view), then multiply.
        BCx_T = BCx.transpose(-1, -2)  # (B, H, S) for each group? Actually, BCx_T[b, s, c] corresponds to original dim.
        # Since BCx is (B, S, 3H), slicing along last dim gives each group.
        # B: first H channels
        B_group = BCx_T[:, :, :H]            # (B, H, S)
        # C: second H channels
        C_group = BCx_T[:, :, H:2*H]        # (B, H, S)
        # x_proj: third H channels
        x_proj_group = BCx_T[:, :, 2*H:]    # (B, H, S)

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = B_group * x_proj_group  # (B, H, S)

        # 3) Left-pad for causal conv: pad 3 zeros (conv_kernel_size=4 => left_pad=3)
        Bx_padded = torch.empty((B, H, S + 3), device=x.device, dtype=dtype)
        BLOCK_S_PAD = 256  # tile along sequence
        grid_pad = (B, H, triton.cdiv(S + 3, BLOCK_S_PAD))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, C=H, S_in=S, S_out=S+3, pad_left=3,
            BLOCK_S=BLOCK_S_PAD,
            num_warps=4, num_stages=2
        )

        # 4) Grouped 1D convolution (depthwise) with groups=H
        # conv_weight: (H, 1, 4) -> treat as (H, 4)
        Wconv = conv_weight.contiguous().to(dtype).view(H, 4)  # (H, 4)
        bias_conv = conv_bias.contiguous().to(dtype)           # (H,)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=dtype)

        grid_conv = (B, H)
        conv1d_depthwise_kernel[grid_conv](
            Bx_padded, Wconv, bias_conv, conv_out,
            B=B, C=H, S=S, S_padded=S+3, K=4,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out
        C_T = C_group.transpose(-1, -2)  # (B, S, H)
        y = C_T * conv_out  # (B, S, H)

        # 6) Final linear projection: F.linear(y, out_proj_weight, out_proj_bias)
        # Implement in Triton
        # y_T is already (B, S, H)
        Wout = out_proj_weight.contiguous().to(dtype)       # (H, H)
        bias_out = out_proj_bias.contiguous().to(dtype)     # (H,)
        output = torch.empty((B, S, H), device=x.device, dtype=dtype)

        BLOCK_H = 128
        grid_out = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        out_proj_matmul_kernel[grid_out](
            y, Wout, bias_out, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
