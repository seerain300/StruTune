import torch
import triton
import triton.language as tl


# -----------------------------
# Kernel 1: Linear projection (in_proj) using matmul style
# Computes: Y[b, s, m] = sum_n X[b, s, n] * W[m, n] + bias[m]
# X shape: (B, S, N), W shape: (M, N), Y shape: (B, S, M)
# -----------------------------
@triton.jit
def in_proj_matmul_kernel(
    X_ptr,        # *f32, shape (B, S, N)
    W_ptr,        # *f32, shape (M, N)
    BIAS_ptr,     # *f32, shape (M,)
    Y_ptr,        # *f32, shape (B, S, M)
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # seq_len
    N: tl.constexpr,   # hidden_size (in_proj input dim)
    M: tl.constexpr,   # 3 * hidden_size (out_proj dim for in_proj)
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

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over N
    # We loop over n from 0 to N-1 (compile-time loop for performance).
    for n in range(0, N):
        # Load X[b, s_offsets, n] -> shape (BLOCK_S,)
        # X layout: ((b * S + s) * N + n)
        x_ptrs = X_ptr + pid_b * S * N + s_offsets * N + n
        x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)
        x_vals = x_vals[:, None]  # (BLOCK_S, 1)

        # Load W[m_offsets, n] -> shape (BLOCK_M,)
        w_ptrs = W_ptr + m_offsets * N + n  # W is (M, N), row-major: W[m, n] = base + m*N + n
        w_vals = tl.load(w_ptrs, mask=m_mask, other=0.0)  # (BLOCK_M,)
        w_vals = w_vals[None, :]  # (1, BLOCK_M)

        # Accumulate outer product
        acc += x_vals * w_vals  # (BLOCK_S, BLOCK_M)

    # Add bias
    bias_vals = tl.load(BIAS_ptr + m_offsets, mask=m_mask, other=0.0)  # (BLOCK_M,)
    acc = acc + bias_vals[None, :]

    # Store to Y
    y_ptrs = Y_ptr + pid_b * S * M + s_offsets[:, None] * M + m_offsets[None, :]
    store_mask = (s_mask[:, None]) & (m_mask[None, :])
    tl.store(y_ptrs, acc, mask=store_mask)


# -----------------------------
# Kernel 2: Left-pad along last dimension (sequence)
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

    # For t < pad_left, write zeros
    is_pad = t_offsets < pad_left

    # Compose pointers
    # IN layout: ((b * C + c) * S_in + t)
    in_ptrs = IN_ptr + pid_b * C * S_in + pid_c * S_in + t_offsets
    # OUT layout: ((b * C + c) * S_out + t)
    out_ptrs = OUT_ptr + pid_b * C * S_out + pid_c * S_out + t_offsets

    # Load values for non-pad positions (t >= pad_left): use t - pad_left
    t_actual = t_offsets - pad_left
    in_ptrs_actual = IN_ptr + pid_b * C * S_in + pid_c * S_in + t_actual
    in_vals = tl.load(in_ptrs_actual, mask=(~is_pad) & t_mask, other=0.0)
    out_vals = tl.where(is_pad & t_mask, 0.0, in_vals)

    tl.store(out_ptrs, out_vals, mask=t_mask)


# -----------------------------
# Kernel 3: Grouped 1D convolution (depthwise) with groups=C
# Input: Bx_padded (B, C, S_padded), Weight: (C, 1, K), Bias: (C,)
# Output: Conv_out (B, C, S)  (no padding in conv, host already padded)
# conv_out[b, c, t] = sum_{k=0..K-1} weight[c, 0, k] * Bx_padded[b, c, t + k] + bias[c]
# -----------------------------
@triton.jit
def conv1d_depthwise_kernel(
    INPUT_ptr,    # *f32, shape (B, C, S_padded)
    WEIGHT_ptr,   # *f32, shape (C, K) where K=4, row-major
    BIAS_ptr,     # *f32, shape (C,)
    OUTPUT_ptr,   # *f32, shape (B, C, S)
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,               # output seq_len (same as input seq_len in padded context)
    S_padded: tl.constexpr,        # S + pad_left
    K: tl.constexpr,               # kernel_size, here 4
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # We'll process all t positions, masked when s > S (we launch grid with S)
    for t in range(0, S):
        acc = 0.0
        # Sum over K taps
        for k in range(0, K):
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
# Kernel 4: Final linear projection (out_proj) using matmul style
# Computes: OUT[b, s, h] = sum_h2 Y_T[b, s, h2] * W_out[h, h2] + bias_out[h]
# Y_T shape: (B, S, H), W_out shape: (H, H), OUT shape: (B, S, H)
# -----------------------------
@triton.jit
def out_proj_matmul_kernel(
    Y_ptr,        # *f32, shape (B, S, H)
    W_ptr,        # *f32, shape (H, H)
    BIAS_ptr,     # *f32, shape (H,)
    OUT_ptr,      # *f32, shape (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_S: tl.constexpr,  # tile size along S
    BLOCK_H: tl.constexpr,  # tile size along H
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    s_mask = s_offsets < S
    h_mask = h_offsets < H

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Reduction over H (note: Y_T is (B, S, H); we reduce over H to produce (B, S, H))
    # Loop over H2 (input hidden dimension), accumulate Y[b, s, h2] * W[h, h2]
    # For each h2, load Y slice and W row h_offsets.
    for h2 in range(0, H):
        # Load Y[b, s_offsets, h2] -> (BLOCK_S,)
        y_ptrs = Y_ptr + pid_b * S * H + s_offsets * H + h2
        y_vals = tl.load(y_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)
        y_vals = y_vals[:, None]  # (BLOCK_S, 1)

        # Load W[h_offsets, h2] -> (BLOCK_H,)
        w_ptrs = W_ptr + h_offsets * H + h2
        w_vals = tl.load(w_ptrs, mask=h_mask, other=0.0)  # (BLOCK_H,)
        w_vals = w_vals[None, :]  # (1, BLOCK_H)

        acc += y_vals * w_vals  # (BLOCK_S, BLOCK_H)

    # Add bias
    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=h_mask, other=0.0)  # (BLOCK_H,)
    acc = acc + bias_vals[None, :]

    # Store to OUT
    out_ptrs = OUT_ptr + pid_b * S * H + s_offsets[:, None] * H + h_offsets[None, :]
    store_mask = (s_mask[:, None]) & (h_mask[None, :])
    tl.store(out_ptrs, acc, mask=store_mask)


# =============================
# ModelNew: Triton-ONLY forward
# =============================
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We will receive weights and biases as parameters in forward call; not stored.

    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Replicates the original 'run' function using Triton kernels for:
        - in_proj linear: F.linear(x, in_proj_weight, in_proj_bias)
        - left-pad for causal conv
        - grouped depthwise 1D conv (groups = hidden_size)
        - out_proj linear: F.linear(y_T, out_proj_weight, out_proj_bias)
        """
        # Ensure dtype and contiguity; use float32 for simplicity
        dtype = torch.float32
        B, S, hidden_size = x.shape

        # 1) Compute B, C, x_proj from in_proj: F.linear(x, in_proj_weight, in_proj_bias)
        #    We use Triton matmul kernel for F.linear: y[b, s, m] = sum_n x[b, s, n] * W[m, n] + bias[m]
        #    in_proj_weight: (M, N) where M=3*hidden_size, N=hidden_size
        M = 3 * hidden_size
        X_in = x.contiguous().to(dtype)
        W_in = in_proj_weight.contiguous().to(dtype)  # (M, N)
        bias_in = in_proj_bias.contiguous().to(dtype)  # (M,)
        BCx = torch.empty((B, S, M), dtype=dtype, device=x.device)

        # Launch in_proj_matmul_kernel
        BLOCK_S = 64
        BLOCK_M = 64
        grid_in = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(S, BLOCK_S))
        in_proj_matmul_kernel[grid_in](
            X_in, W_in, bias_in, BCx,
            B=B, S=S, N=hidden_size, M=M,
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
        )

        # Split BCx into B, C, x_proj
        # Each has shape (B, hidden_size, S)
        # We can compute views by taking slices: B=first hidden_size, C=middle hidden_size, x_proj=last hidden_size
        B_tensor = BCx[:, :, :hidden_size]    # (B, S, H)
        C_tensor = BCx[:, :, hidden_size:2*hidden_size]  # (B, S, H)
        x_proj = BCx[:, :, 2*hidden_size:]    # (B, S, H)

        # 2) Element-wise gating: Bx = B * x_proj
        # This is simple elementwise multiply; we’ll keep it in PyTorch for now (no Triton kernel needed in host).
        Bx = B_tensor * x_proj

        # Transpose for conv1d: (B, C, S)
        # We need Bx_t shape (B, C, S) where C=hidden_size. We already have (B, S, C) if we use x_proj as channels.
        # Here, C_tensor is (B, S, H); to use conv1d, we want (B, H, S). So we transpose:
        Bx_t = C_tensor.transpose(-1, -2).contiguous()  # (B, H, S)

        # 3) Causal left-pad: pad 3 elements on the left (conv_kernel_size=4 -> pad=3)
        left_pad = 3
        S_padded = S + left_pad
        Bx_padded = torch.empty((B, hidden_size, S_padded), dtype=dtype, device=x.device)

        # Launch pad_left_kernel
        BLOCK_S_pad = 128
        grid_pad = (B, hidden_size, triton.cdiv(S_padded, BLOCK_S_pad))
        pad_left_kernel[grid_pad](
            Bx_t, Bx_padded,
            B=B, C=hidden_size, S_in=S, S_out=S_padded,
            pad_left=left_pad,
            BLOCK_S=BLOCK_S_pad,
        )

        # 4) Grouped depthwise 1D conv: groups=C=hidden_size
        # conv_weight shape: (C, 1, K) where K=4; we can view as (C, K) by dropping the singleton
        conv_weight_reshaped = conv_weight.reshape(hidden_size, conv_weight.shape[2]).contiguous()  # (C, K)
        conv_bias_vec = conv_bias.contiguous()  # (C,)

        conv_out = torch.empty((B, hidden_size, S), dtype=dtype, device=x.device)

        # Launch conv1d_depthwise_kernel
        grid_conv = (B, hidden_size)
        conv1d_depthwise_kernel[grid_conv](
            Bx_padded, conv_weight_reshaped, conv_bias_vec, conv_out,
            B=B, C=hidden_size, S=S, S_padded=S_padded, K=4,
        )

        # 5) Output gating: y = C_tensor * conv_out
        # C_tensor: (B, hidden_size, S) = (B, C, S) after transposing back
        y = C_tensor * conv_out

        # 6) Transpose y back to (B, S, hidden_size) and do final linear projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, hidden_size)

        # Final linear projection
        out = torch.empty((B, S, hidden_size), dtype=dtype, device=x.device)

        # Use Triton matmul for final linear
        H = hidden_size
        W_out = out_proj_weight.contiguous().to(dtype)  # (H, H)
        bias_out = out_proj_bias.contiguous().to(dtype)  # (H,)
        BLOCK_S_out = 64
        BLOCK_H_out = 64
        grid_out = (B, triton.cdiv(H, BLOCK_H_out), triton.cdiv(S, BLOCK_S_out))
        out_proj_matmul_kernel[grid_out](
            y_T, W_out, bias_out, out,
            B=B, S=S, H=H,
            BLOCK_S=BLOCK_S_out, BLOCK_H=BLOCK_H_out,
        )

        return out


def run(*args):
    return ModelNew()(*args)
