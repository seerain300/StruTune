import torch
import triton
import triton.language as tl


# -----------------------------
# Triton kernel: Linear projection F.linear(x, W, bias)
# x: (B, S, H), W: (M, H), bias: (M,), output: (B, S, M)
# We use 2D tiles along M and S; static loop over H.
# -----------------------------
@triton.jit
def in_proj_matmul_kernel(
    X_ptr,           # *f32, shape (B, S, H)
    W_ptr,           # *f32, shape (M, H)
    BIAS_ptr,        # *f32, shape (M,)
    OUT_ptr,         # *f32, shape (B, S, M)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Masks
    s_mask = offs_s < S
    m_mask = offs_m < M

    # Accumulator
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over H: W is (M, H), X is (B, S, H)
    for h in range(0, H):
        # Load X tile for (b, offs_s, h)
        x_ptrs = X_ptr + pid_b * (S * H) + offs_s * H + h
        x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)

        # Load W tile for (offs_m, h)
        w_ptrs = W_ptr + offs_m * H + h                    # (BLOCK_M,)
        w_vals = tl.load(w_ptrs, mask=m_mask, other=0.0)  # (BLOCK_M,)

        # Outer product: (BLOCK_S, 1) * (1, BLOCK_M) -> (BLOCK_S, BLOCK_M)
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_vals = tl.load(BIAS_ptr + offs_m, mask=m_mask, other=0.0)  # (BLOCK_M,)
    acc = acc + bias_vals[None, :]

    # Store to OUT: (B, S, M)
    out_ptrs = OUT_ptr + pid_b * (S * M) + (offs_s[:, None] * M) + offs_m[None, :]
    store_mask = (s_mask[:, None]) & (m_mask[None, :])
    tl.store(out_ptrs, acc, mask=store_mask)


# -----------------------------
# Triton kernel: Left-pad along sequence dimension
# Input: BCx (B, M, S), Output: Padded (B, M, S_padded)
# pad_left is the number of zeros added at the beginning.
# -----------------------------
@triton.jit
def left_pad_kernel(
    IN_ptr,          # *f32, shape (B, M, S)
    OUT_ptr,         # *f32, shape (B, M, S_padded)
    B: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    pad_left: tl.constexpr,
    S_padded: tl.constexpr,  # S + pad_left
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_sp = tl.program_id(2)  # iterate over S_padded

    # If pid_sp < pad_left: write zeros; else copy from IN[b, m, pid_sp - pad_left]
    if pid_sp < pad_left:
        out_val = 0.0
    else:
        s_src = pid_sp - pad_left
        in_ptrs = IN_ptr + pid_b * (M * S) + pid_m * S + s_src
        in_val = tl.load(in_ptrs)
        out_val = in_val

    out_ptrs = OUT_ptr + pid_b * (M * S_padded) + pid_m * S_padded + pid_sp
    tl.store(out_ptrs, out_val)


# -----------------------------
# Triton kernel: Grouped 1D conv with groups=B
# Input: Bx_padded (B, M, S_padded), Weight: (M, 1, K), Bias: (M,)
# Output: Conv_out (B, M, S)  (no conv padding needed; host already padded)
# conv_out[b, m, t] = sum_{k=0..K-1} weight[m, 0, k] * Bx_padded[b, m, t + k] + bias[m]
# We implement one program per (b, m) and vectorize over t.
# For generality, we use a static loop over K (here K is passed as constexpr).
# -----------------------------
@triton.jit
def conv1d_depthwise_groupsB_kernel(
    INPUT_ptr,   # *f32, shape (B, M, S_padded)
    WEIGHT_ptr,  # *f32, shape (M, K), row-major across K
    BIAS_ptr,    # *f32, shape (M,)
    OUTPUT_ptr,  # *f32, shape (B, M, S)
    B: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    S_padded: tl.constexpr,
    K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Vector of output positions t
    t_offsets = tl.arange(0, 1024)  # cap to 1024; will mask beyond S
    t_mask = t_offsets < S

    # Accumulator for this (b, m)
    acc = tl.zeros((1024,), dtype=tl.float32)

    # Sum over K taps; K is small (e.g., 4), so loop is fine
    for k in range(0, K):
        # Load weights for this m and k
        w_index = pid_m * K + k
        w_val = tl.load(WEIGHT_ptr + w_index)  # scalar

        # Load input along sequence for each t: position = t + k
        pos_offsets = t_offsets + k
        pos_mask = (pos_offsets < S_padded) & (t_mask)
        in_ptrs = INPUT_ptr + pid_b * (M * S_padded) + pid_m * S_padded + pos_offsets
        in_vals = tl.load(in_ptrs, mask=pos_mask, other=0.0)  # (1024,)
        acc += in_vals * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_m)
    acc += bias_val

    # Store to output at t_offsets, masked by t_mask
    out_ptrs = OUTPUT_ptr + pid_b * (M * S) + pid_m * S + t_offsets
    tl.store(out_ptrs, acc, mask=t_mask)


# -----------------------------
# Triton kernel: Final linear projection (out_proj)
# Input: Y_T (B, S, H), W_out (H, H), bias_out (H)
# Output: OUT (B, S, H)
# We implement a simple grid over (B, S, H) tiles along H2 dimension and reduce over H2.
# Because H is small in our setup, we can keep it simple. Here we use a single tile over H2=H.
# -----------------------------
@triton.jit
def out_proj_matmul_kernel(
    Y_ptr,          # *f32, shape (B, S, H) (transposed from original)
    Wout_ptr,       # *f32, shape (H, H)
    Bout_ptr,       # *f32, shape (H,) bias
    OUT_ptr,        # *f32, shape (B, S, H)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Reduce over h2 in H; we can do a simple loop since H is small.
    for h2 in range(0, H):
        # Load y[b, s, h2]
        y_ptrs = Y_ptr + pid_b * (S * H) + pid_s * H + h2
        y_val = tl.load(y_ptrs)  # scalar

        # Load Wout[h, h2] for our h_offsets
        w_ptrs = Wout_ptr + h_offsets * H + h2
        w_vals = tl.load(w_ptrs, mask=h_mask, other=0.0)  # (BLOCK_H,)

        acc += y_val * w_vals

    # Add bias_out
    bias_vals = tl.load(Bout_ptr + h_offsets, mask=h_mask, other=0.0)
    acc += bias_vals

    # Store output
    out_ptrs = OUT_ptr + pid_b * (S * H) + pid_s * H + h_offsets
    tl.store(out_ptrs, acc, mask=h_mask)


# -----------------------------
# ModelNew: entry point
# -----------------------------
class ModelNew(torch.nn.Module):
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
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        M = 3 * H

        # 1) Initial linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # Implement with Triton
        x_in = x.contiguous().to(dtype)                   # (B, S, H)
        W_in = in_proj_weight.contiguous().to(dtype)     # (M, H)
        bias_in = in_proj_bias.contiguous().to(dtype)    # (M,)
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)

        BLOCK_S = 128
        BLOCK_M = 128
        grid_in = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(S, BLOCK_S))
        in_proj_matmul_kernel[grid_in](
            x_in, W_in, bias_in, BCx,
            B=B, S=S, H=H, M=M,
            BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: split BCx into (B, C, x_proj) along last dim (M=3H), then Bx = B * x_proj
        # Shapes: B: (B, S, H), C: (B, S, H), x_proj: (B, S, H)
        B_mat = BCx[:, :, :H]
        C_mat = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        Bx = B_mat * x_proj  # elementwise multiply, (B, S, H)

        # 3) Left-pad along sequence for causal conv (kernel_size=4 => pad_left=3)
        pad_left = conv_weight.shape[2] - 1
        S_padded = S + pad_left
        Bx_padded = torch.empty((B, M, S_padded), device=device, dtype=dtype)

        grid_pad = (B, M, S_padded)
        left_pad_kernel[grid_pad](
            BCx, Bx_padded,
            B=B, M=M, S=S, pad_left=pad_left, S_padded=S_padded,
            num_warps=1, num_stages=1
        )

        # 4) Grouped 1D conv with groups=B: conv_out[b, m, t] = sum_{k=0..3} weight[m, 0, k] * Bx_padded[b, m, t+k] + bias[m]
        conv_weight_2d = conv_weight.view(H, 4).contiguous().to(dtype)   # (H, K)
        conv_bias_2d = conv_bias.contiguous().to(dtype)                  # (H,)
        conv_out = torch.empty((B, M, S), device=device, dtype=dtype)

        grid_conv = (B, M)
        # Launch with num_warps=1; K=4 is small, single program per (b, m) vectorizing over S
        conv1d_depthwise_groupsB_kernel[grid_conv](
            Bx_padded, conv_weight_2d, conv_bias_2d, conv_out,
            B=B, M=M, S=S, S_padded=S_padded, K=4,
            num_warps=1, num_stages=1
        )

        # 5) Output gating: y = C * conv_out, where C is the second group (B, S, H) from BCx
        # C is (B, S, H), conv_out is (B, M, S); we need to use only channels m in [H, 2H), i.e., the C group.
        # But conv_out is computed across all m; here, PyTorch's original code uses conv_out per channel; with groups=B,
        # each (b, m) convolves its own slice. We gate with C matrix for all m? The original splits (B, C, x_proj) and uses C only after conv.
        # In our code, conv_out shape is (B, M, S). The gating y = C * conv_out likely means using C corresponding to channels in [H, 2H).
        # To match exactly, we reconstruct C as BCx[:, :, H:2H] and multiply elementwise: y = C * conv_out.
        # Note: conv_out has channels dimension M, but C is (B, S, H). We need to align dims. Given original code's gating y = C * conv_out,
        # and C is (B, hidden, seq), while conv_out is (B, hidden, seq). So here conv_out's channels should correspond to H.
        # In our setup, conv_weight is (H, 1, 4), bias (H,), so conv_out is (B, H, S). Earlier we used conv_weight (H,1,4) but computed over M channels;
        # to match original semantics, we should compute conv_out per channel H and gate with C (channels H). The previous approach computed over M (3H),
        # which likely deviates from original behavior. We will adjust conv_out to be (B, H, S) by using conv_weight (H,1,4), conv_bias (H,).
        # Let's correct conv1d to produce (B, H, S).

        # Recompute conv_out correctly: conv_weight is (H,1,4) -> conv_out (B,H,S)
        conv_out_H = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_conv_H = (B, H)
        conv1d_depthwise_groupsB_kernel[grid_conv_H](
            Bx_padded, conv_weight.view(H, 4).contiguous().to(dtype), conv_bias.contiguous().to(dtype), conv_out_H,
            B=B, M=H, S=S, S_padded=S_padded, K=4,
            num_warps=1, num_stages=1
        )

        # Gating with C: C is (B, S, H)
        y = C_mat * conv_out_H  # (B, S, H)

        # 6) Transpose back to (B, S, H): y already (B, S, H)
        y_T = y  # no transpose needed; we want (B, S, H) for final linear

        # 7) Final linear projection: out = F.linear(y_T, out_proj_weight, out_proj_bias)
        # Implement with Triton. y_T shape (B, S, H), W_out (H, H), bias_out (H,)
        W_out = out_proj_weight.contiguous().to(dtype)  # (H, H)
        bias_out = out_proj_bias.contiguous().to(dtype) # (H,)
        out = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_matmul_kernel[grid_out](
            y_T, W_out, bias_out, out,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
