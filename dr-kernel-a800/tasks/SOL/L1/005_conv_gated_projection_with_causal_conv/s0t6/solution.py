import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: in_proj = F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (M=3H, H), in_proj_bias: (M=3H,)
# out: BCx (B, S, M)
@triton.jit
def in_proj_kernel(
    X_ptr,          # *f32, (B, S, H)
    W_ptr,          # *f32, (M, H) where M=3*H
    BIAS_ptr,       # *f32, (M,)
    OUT_ptr,        # *f32, (B, S, M)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,                 # 3*H
    BLOCK_S: tl.constexpr,           # tile along S
    BLOCK_M: tl.constexpr,           # tile along M
):
    pid_b = tl.program_id(0)
    pid_s_tile = tl.program_id(1)
    pid_m_tile = tl.program_id(2)

    s_offsets = pid_s_tile * BLOCK_S + tl.arange(0, BLOCK_S)        # [BLOCK_S]
    m_offsets = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)        # [BLOCK_M]
    s_mask = s_offsets < S
    m_mask = m_offsets < M

    # Accumulator for OUT[b, s_offsets, m_offsets]
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over H
    for h in tl.static_range(0, H):
        # Load X[b, s, h] for all s in tile
        x_index = pid_b * S * H + s_offsets * H + h
        x_vec = tl.load(X_ptr + x_index, mask=s_mask, other=0.0)    # [BLOCK_S]

        # Load W[m, h] for all m in tile
        w_index = m_offsets * H + h                                   # since W is (M, H), row-major
        w_vec = tl.load(W_ptr + w_index, mask=m_mask, other=0.0)     # [BLOCK_M]

        # Outer product accumulate: acc[s, m] += x_vec[s] * w_vec[m]
        acc += x_vec[:, None] * w_vec[None, :]

    # Add bias: BIAS[m_offsets]
    bias_vec = tl.load(BIAS_ptr + m_offsets, mask=m_mask, other=0.0)  # [BLOCK_M]
    acc += bias_vec[None, :]

    # Store to OUT[b, s, m]
    out_index = pid_b * S * M + s_offsets[:, None] * M + m_offsets[None, :]
    store_mask = s_mask[:, None] & m_mask[None, :]
    tl.store(OUT_ptr + out_index, acc, mask=store_mask)


# Triton kernel: left-pad along sequence dimension (causal pad)
# IN: (B, M, S), OUT: (B, M, S+K-1). Pad value = 0.0.
# We implement pad on left by writing zeros to t < pad_left and copying IN[:, :, t-pad_left] to OUT[:, :, t].
# Here, M can be any, but in our use it is 3*H. K is kernel_size (here 4).
@triton.jit
def left_pad_kernel(
    IN_ptr,         # *f32, (B, M, S)
    OUT_ptr,        # *f32, (B, M, S_out)
    B: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,                # kernel_size (4)
    S_out: tl.constexpr,            # S + K - 1
    BLOCK_M: tl.constexpr,          # tile along M
    BLOCK_S: tl.constexpr,          # tile along S_out
):
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_s_tile = tl.program_id(2)

    m_offsets = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    s_offsets = pid_s_tile * BLOCK_S + tl.arange(0, BLOCK_S)    # [BLOCK_S]
    m_mask = m_offsets < M
    s_mask = s_offsets < S_out

    # For each s_out, either it maps to s = s_out - (K-1) if s_out >= K-1, else pad with zeros.
    # We compute out index and input index accordingly.
    for t in tl.static_range(0, BLOCK_S):                        # process one s-offset at a time for simplicity
        s_out_idx = pid_s_tile * BLOCK_S + t
        if s_out_idx < S_out:
            s_from = s_out_idx - (K - 1)                        # pad left by K-1
            # Valid if s_from in [0, S)
            valid = s_from >= 0 and s_from < S

            out_index = pid_b * M * S_out + m_offsets * S_out + s_out_idx
            in_index = pid_b * M * S + m_offsets * S + s_from

            # Load input with mask (only if valid) and store to output
            in_vec = tl.load(IN_ptr + in_index, mask=m_mask & valid, other=0.0)
            tl.store(OUT_ptr + out_index, in_vec, mask=m_mask)


# Triton kernel: grouped causal 1D convolution with groups = C_in = M
# IN: (B, M, S_in) where M=3*H, S_in=S+K-1 (padded), weight: (M, 1, K), bias: (M,)
# OUT: (B, M, S_out) where S_out=S. This is causal conv: output length S_out=S.
@triton.jit
def conv1d_groups_kernel(
    IN_ptr,         # *f32, (B, M, S_in) = padded Bx (B, 3H, S+K-1)
    W_ptr,          # *f32, (M, K) row-major
    BIAS_ptr,       # *f32, (M,)
    OUT_ptr,        # *f32, (B, M, S_out) = (B, 3H, S)
    B: tl.constexpr,
    M: tl.constexpr,
    S_in: tl.constexpr,               # S+K-1
    K: tl.constexpr,                  # 4
    S_out: tl.constexpr,              # S
    BLOCK_S: tl.constexpr,            # tile along S_out
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s_tile = tl.program_id(2)

    s_offsets = pid_s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_out

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Sum over K taps
    for k in tl.static_range(0, K):
        # IN index for all s_offsets: b*M*S_in + m*S_in + (s_out + k)
        in_index = pid_b * M * S_in + pid_m * S_in + (s_offsets + k)
        inp_vec = tl.load(IN_ptr + in_index, mask=s_mask, other=0.0)

        # Weight for this m and k: W[m, k]
        w_index = pid_m * K + k
        w_val = tl.load(W_ptr + w_index)
        acc += inp_vec * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_m)
    acc += bias_val

    # Store OUT[b, m, s_offsets]
    out_index = pid_b * M * S_out + pid_m * S_out + s_offsets
    tl.store(OUT_ptr + out_index, acc, mask=s_mask)


# Triton kernel: final linear projection (out_proj)
# Y: (B, S, H), W: (H, H), Bias: (H,)
# OUT[b, s, h] = sum_{h2=0..H-1} Y[b, s, h2] * W[h2, h] + Bias[h]
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
    for h2 in tl.static_range(0, H):
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
        # No parameters; we rely on inputs to provide weights/bias

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H), in_proj_bias: (3H,)
        conv_weight: (H, 1, 4), conv_bias: (H,)
        out_proj_weight: (H, H), out_proj_bias: (H,)
        All tensors are expected to be float32 and contiguous on CUDA device.
        """
        assert x.is_cuda, "ModelNew.forward requires CUDA tensors for Triton kernels."
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
        M = 3 * H
        K = conv_weight.shape[2]
        S_in = S + K - 1
        S_out = S

        # 1) Initial linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # Launch Triton kernel
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)

        # Grid: (B, tiles over S, tiles over M)
        BLOCK_S_IN = 128
        BLOCK_M_IN = 64
        grid_in = (B, triton.cdiv(S, BLOCK_S_IN), triton.cdiv(M, BLOCK_M_IN))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, M=M,
            BLOCK_S=BLOCK_S_IN, BLOCK_M=BLOCK_M_IN,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj along the last dimension (channels) of size H:
        # B: BCx[:, :, :H], C: BCx[:, :, H:2H], x_proj: BCx[:, :, 2H:3H]
        B_tensor = BCx[:, :, :H].contiguous()
        C_tensor = BCx[:, :, H:2 * H].contiguous()
        x_proj = BCx[:, :, 2 * H:].contiguous()

        # 3) Element-wise gating: Bx = B * x_proj, shape (B, H, S)
        Bx = (B_tensor * x_proj).contiguous()  # (B, H, S)
        Bx = Bx.transpose(1, 2).contiguous()   # transpose to (B, S, H) for conv input

        # 4) Left-pad for causal conv: pad left by K-1
        Bx_padded = torch.empty((B, M, S_in), device=device, dtype=dtype)  # M=3H, S_in=S+K-1
        # Launch Triton kernel for left pad: grid (B, tiles over M, tiles over S_out)
        BLOCK_M_PAD = 64
        BLOCK_S_PAD = 128
        grid_pad = (B, triton.cdiv(M, BLOCK_M_PAD), triton.cdiv(S_in, BLOCK_S_PAD))
        left_pad_kernel[grid_pad](
            Bx, Bx_padded,
            B=B, M=M, S=S, K=K, S_out=S_in,
            BLOCK_M=BLOCK_M_PAD, BLOCK_S=BLOCK_S_PAD,
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal 1D convolution with groups=M
        conv_out = torch.empty((B, M, S_out), device=device, dtype=dtype)  # (B, 3H, S)
        grid_conv = (B, triton.cdiv(M, 1), triton.cdiv(S_out, 128))        # one program per m, tile S
        conv1d_groups_kernel[grid_conv](
            Bx_padded, conv_weight.reshape(M, K), conv_bias,
            conv_out,
            B=B, M=M, S_in=S_in, K=K, S_out=S_out,
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C * conv_out, shape (B, H, S)
        # conv_out: (B, 3H, S), C: (B, H, S)
        C_tensor = C_tensor.transpose(1, 2).contiguous()   # (B, S, H)
        y = (C_tensor * conv_out[:, :H, :]).contiguous()   # gate with the first H channels (size H)

        # 7) Final linear projection: y -> out_proj(y)
        # y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H_OUT = 128
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_OUT))
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_OUT,
            num_warps=4, num_stages=2
        )

        return output


# -----------------------------
# Helper to generate dummy inputs (not required by the evaluator, but useful for testing)
# -----------------------------
def _generate_dummy_inputs(batch_size, seq_len, hidden_size):
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    # in_proj_weight: (3*hidden_size, hidden_size)
    in_proj_weight = torch.randn(3 * hidden_size, hidden_size, device='cuda', dtype=torch.float32)
    in_proj_bias = torch.randn(3 * hidden_size, device='cuda', dtype=torch.float32)
    # conv_weight: (hidden_size, 1, 4), conv_bias: (hidden_size,)
    conv_weight = torch.randn(hidden_size, 1, 4, device='cuda', dtype=torch.float32)
    conv_bias = torch.randn(hidden_size, device='cuda', dtype=torch.float32)
    # out_proj_weight: (hidden_size, hidden_size), out_proj_bias: (hidden_size,)
    out_proj_weight = torch.randn(hidden_size, hidden_size, device='cuda', dtype=torch.float32)
    out_proj_bias = torch.randn(hidden_size, device='cuda', dtype=torch.float32)
    return x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias


# -----------------------------
# Example usage:
# x, in_proj_w, in_proj_b, conv_w, conv_b, out_proj_w, out_proj_b = _generate_dummy_inputs(2, 4096, 128)
# model = ModelNew().cuda()
# out = model(x, in_proj_w, in_proj_b, conv_w, conv_b, out_proj_w, out_proj_b)
# print(out.shape)
# -----------------------------


def run(*args):
    return ModelNew()(*args)
