import torch
import triton
import triton.language as tl

# 1) Triple linear projection kernel: computes out[B, S, H] = x @ W^T + b
#    We launch it three times (W0 for B, W1 for C, W2 for x_proj).
@triton.jit
def linear_bsh_kernel(
    x_ptr,      # *f32, (B, S, H), contiguous
    W_ptr,      # *f32, (H, H), contiguous
    b_ptr,      # *f32, (H,), contiguous
    out_ptr,    # *f32, (B, S, H), contiguous
    B: tl.int32, S: tl.int32, H: tl.int32,
    x_stride0, x_stride1, x_stride2,
    W_stride0, W_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)  # batch dimension
    h_out = tl.program_id(1)  # output channel (hidden dimension)
    s_block = tl.program_id(2)  # tile along sequence
    s_start = s_block * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Accumulator for out[b, s_offsets, h_out]
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over K dimension (input hidden size) in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load x[b, s, k] for all s in tile and k chunk
        x_ptrs = x_ptr + b_id * x_stride0 + s_offsets[:, None] * x_stride1 + k_offsets[None, :] * x_stride2
        x_mask = s_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # shape [BLOCK_S, BLOCK_K]

        # Load W[h_out, k] for k chunk
        w_ptrs = W_ptr + h_out * W_stride0 + k_offsets * W_stride1
        w_vals = tl.load(w_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # Outer product accumulation: acc[s] += sum_k x[b, s, k] * W[h_out, k]
        # Broadcast x_vals [BLOCK_S, BLOCK_K] and w_vals [BLOCK_K] -> [BLOCK_S, BLOCK_K]
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)

    # Add bias
    b_val = tl.load(b_ptr + h_out)
    acc += b_val

    # Store result to out[b, s_offsets, h_out]
    out_ptrs = out_ptr + b_id * out_stride0 + s_offsets * out_stride1 + h_out * out_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 2) Elementwise gating: out = A * B, both (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    A_ptr, B_ptr, Out_ptr,
    B: tl.int32, S: tl.int32, H: tl.int32,
    A_stride0, A_stride1, A_stride2,
    B_stride0, B_stride1, B_stride2,
    Out_stride0, Out_stride1, Out_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    a_ptrs = A_ptr + b_id * A_stride0 + s_offsets * A_stride1 + h_id * A_stride2
    b_ptrs = B_ptr + b_id * B_stride0 + s_offsets * B_stride1 + h_id * B_stride2
    out_ptrs = Out_ptr + b_id * Out_stride0 + s_offsets * Out_stride1 + h_id * Out_stride2

    a = tl.load(a_ptrs, mask=s_mask, other=0.0)
    b = tl.load(b_ptrs, mask=s_mask, other=0.0)
    out = a * b
    tl.store(out_ptrs, out, mask=s_mask)

# 3) Grouped causal 1D convolution: out[B, H, S] from in[B, S, H], convW(H, H, 4), bias(H), groups=H
#    conv_out[b, c, t] = sum_{k=0..3} in[b, c, t + k - 1] * convW[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    in_ptr,      # *f32, (B, S, H), contiguous
    convW_ptr,   # *f32, (H, H, 4), contiguous
    convB_ptr,   # *f32, (H,), contiguous
    out_ptr,     # *f32, (B, H, S), contiguous
    B: tl.int32, S: tl.int32, H: tl.int32,
    in_stride0, in_stride1, in_stride2,
    convW_stride0, convW_stride1, convW_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)  # output channel equals group id
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S

    # We will compute for positions t in [s_start, s_start+BLOCK_S-1]
    # The causal input index is t + k - 1; for k=0..3, this is t-1, t, t+1, t+2
    # We need to mask invalid indices.

    # Load conv weights for this group c_id: [w0, w1, w2, w3]
    w0 = tl.load(convW_ptr + c_id * convW_stride0 + c_id * convW_stride1 + 0 * convW_stride2)
    w1 = tl.load(convW_ptr + c_id * convW_stride0 + c_id * convW_stride1 + 1 * convW_stride2)
    w2 = tl.load(convW_ptr + c_id * convW_stride0 + c_id * convW_stride1 + 2 * convW_stride2)
    w3 = tl.load(convW_ptr + c_id * convW_stride0 + c_id * convW_stride1 + 3 * convW_stride2)

    # Add bias
    bias = tl.load(convB_ptr + c_id)

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for t_idx in range(0, BLOCK_S):
        t = s_start + t_idx
        # Check if t within S
        t_valid = t < S

        # Compute input indices for each k with left padding
        t_m1 = t - 1
        t0  = t
        t1  = t + 1
        t2  = t + 2

        m0 = (t_m1 >= 0) & t_valid
        m1 = t0 < S
        m2 = (t1 < S) & t_valid
        m3 = (t2 < S) & t_valid

        # Load in[b, c_id, idx_k] with masks
        in0 = tl.load(in_ptr + b_id * in_stride0 + c_id * in_stride1 + t_m1 * in_stride2, mask=m0, other=0.0)
        in1 = tl.load(in_ptr + b_id * in_stride0 + c_id * in_stride1 + t0 * in_stride2,  mask=m1, other=0.0)
        in2 = tl.load(in_ptr + b_id * in_stride0 + c_id * in_stride1 + t1 * in_stride2,  mask=m2, other=0.0)
        in3 = tl.load(in_ptr + b_id * in_stride0 + c_id * in_stride1 + t2 * in_stride2,  mask=m3, other=0.0)

        # Accumulate
        acc[t_idx] = in0 * w0 + in1 * w1 + in2 * w2 + in3 * w3 + bias

    # Store to out[b, c_id, t]
    out_ptrs = out_ptr + b_id * out_stride0 + c_id * out_stride1 + (s_start + tl.arange(0, BLOCK_S)) * out_stride2
    tl.store(out_ptrs, acc, mask=(s_start + tl.arange(0, BLOCK_S)) < (S))

# 4) Final linear projection: y[B, S, H] = conv_out[B, H, S] @ out_proj_weight^T + out_proj_bias
#    Implemented via an elementwise outer-product accumulation over K=H chunks.
@triton.jit
def final_linear_bsh_kernel(
    yin_ptr,     # *f32, (B, H, S), contiguous
    Wout_ptr,    # *f32, (H, H), contiguous
    bout_ptr,    # *f32, (H,), contiguous
    yout_ptr,    # *f32, (B, S, H), contiguous
    B: tl.int32, S: tl.int32, H: tl.int32,
    yin_stride0, yin_stride1, yin_stride2,
    Wout_stride0, Wout_stride1,
    yout_stride0, yout_stride1, yout_stride2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)
    h_out = tl.program_id(1)  # output channel
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load yin[b, k, s] for all s in tile and k chunk: yin shape (B, H, S)
        y_ptrs = yin_ptr + b_id * yin_stride0 + k_offsets[:, None] * yin_stride1 + s_offsets[None, :] * yin_stride2
        y_mask = k_mask[:, None] & s_mask[None, :]
        y_vals = tl.load(y_ptrs, mask=y_mask, other=0.0)  # [BLOCK_K, BLOCK_S]

        # Load Wout[h_out, k] for k chunk
        w_ptrs = Wout_ptr + h_out * Wout_stride0 + k_offsets * Wout_stride1
        w_vals = tl.load(w_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Outer product accumulation: acc[s] += sum_k yin[b, k, s] * Wout[h_out, k]
        acc += tl.sum(y_vals * w_vals[None, :], axis=1)

    # Add bias
    b_val = tl.load(bout_ptr + h_out)
    acc += b_val

    # Store to yout[b, s, h_out]
    yout_ptrs = yout_ptr + b_id * yout_stride0 + s_offsets * yout_stride1 + h_out * yout_stride2
    tl.store(yout_ptrs, acc, mask=s_mask)

# Host function: ModelNew.forward, uses Triton kernels only
class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H)
        # in_proj_weight: (3*H, H)
        # in_proj_bias: (3*H,)
        # conv_weight: (H, H, 4) -> groups=H, kernel_size=4
        # conv_bias: (H,)
        # out_proj_weight: (H, H)
        # out_proj_bias: (H,)
        device = x.device
        dtype = x.dtype

        B, S, H = x.shape
        assert in_proj_weight.shape == (3 * H, H), "in_proj_weight must have shape (3*hidden, hidden)"
        assert in_proj_bias.shape == (3 * H,), "in_proj_bias must have shape (3*hidden,)"
        assert conv_weight.shape == (H, H, 4), "conv_weight must have shape (hidden, hidden, 4)"
        assert conv_bias.shape == (H,), "conv_bias must have shape (hidden,)"
        assert out_proj_weight.shape == (H, H), "out_proj_weight must have shape (hidden, hidden)"
        assert out_proj_bias.shape == (H,), "out_proj_bias must have shape (hidden,)"

        # 1) Triple linear projection
        # Prepare outputs (B, S, H) for B, C, x_proj
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Weight slices for three groups
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()

        W1 = in_proj_weight[H:2 * H, :].contiguous()
        b1 = in_proj_bias[H:2 * H].contiguous()

        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b2 = in_proj_bias[2 * H:3 * H].contiguous()

        # Launch kernels for each output
        # Grid: (B, H, tiles of S)
        BLOCK_S = 128
        grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        linear_bsh_kernel[grid](
            x, W0, b0, B_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )
        linear_bsh_kernel[grid](
            x, W1, b1, C_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )
        linear_bsh_kernel[grid](
            x, W2, b2, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out (B, H, S)
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # output is (B,H,S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B,H,S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 5) Final linear projection to output (B, S, H)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        # out_proj_weight is (H, H), out_proj_bias (H)
        Wout = out_proj_weight.contiguous()  # (H, H)
        bout = out_proj_bias.contiguous()    # (H,)
        grid_final = (B, H, (S + 128 - 1) // 128)
        final_linear_bsh_kernel[grid_final](
            y, Wout, bout, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            Wout.stride(0), Wout.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
