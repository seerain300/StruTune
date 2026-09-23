import torch
import triton
import triton.language as tl

# 1) Triple linear projection: out[B, S, H] = x @ W^T + b
#    We slice in_proj_weight into three (H, H) groups (W0, W1, W2) and run this kernel three times.
@triton.jit
def linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H), contiguous
    W_ptr,         # *f32, (H, H), contiguous
    b_ptr,         # *f32, (H,), contiguous
    out_ptr,       # *f32, (B, S, H), contiguous
    B: tl.int32, S: tl.int32, H: tl.int32,
    x_stride0, x_stride1, x_stride2,
    W_stride0, W_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)          # batch dimension
    h_id = tl.program_id(1)          # output channel h
    s_block = tl.program_id(2)       # tile over S
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load x[b, s, k] for all s in tile and k chunk
        x_ptrs = x_ptr + b_id * x_stride0 + s_offsets[None, :] * x_stride1 + k_offsets[:, None] * x_stride2
        x_mask = s_mask[None, :] & k_mask[:, None]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_K, BLOCK_S]

        # Load W[h, k] for k chunk
        w_ptrs = W_ptr + h_id * W_stride0 + k_offsets * W_stride1
        w_vals = tl.load(w_ptrs, mask=k_mask, other=0.0)   # [BLOCK_K]

        # Accumulate: acc += sum_k (x_vals[k, :] * w_vals[k])
        # x_vals: [K, S], w_vals: [K] -> broadcast multiply and sum over K
        acc += tl.sum(x_vals * w_vals[:, None], axis=0)

    # Add bias
    bias_val = tl.load(b_ptr + h_id)
    acc += bias_val

    # Store out[b, h, s]
    out_ptrs = out_ptr + b_id * out_stride0 + h_id * out_stride1 + s_offsets * out_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 2) Element-wise gating: Bx = B * X (B,S,H)
@triton.jit
def elemwise_mul_bsh_kernel(
    in0_ptr, in1_ptr, out_ptr,  # *f32
    B: tl.int32, S: tl.int32, H: tl.int32,
    in0_stride0, in0_stride1, in0_stride2,
    in1_stride0, in1_stride1, in1_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    a_ptrs = in0_ptr + b_id * in0_stride0 + h_id * in0_stride1 + s_offsets * in0_stride2
    b_ptrs = in1_ptr + b_id * in1_stride0 + h_id * in1_stride1 + s_offsets * in1_stride2
    out_ptrs = out_ptr + b_id * out_stride0 + h_id * out_stride1 + s_offsets * out_stride2

    a = tl.load(a_ptrs, mask=s_mask, other=0.0)
    b = tl.load(b_ptrs, mask=s_mask, other=0.0)
    c = a * b

    tl.store(out_ptrs, c, mask=s_mask)

# 3) Grouped causal 1D convolution:
#    Input Bx: (B, S, H), Weight convW: (H, H, 4), Bias convB: (H,)
#    Output conv_out: (B, H, S)
#    conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t + k - 1] * convW[c, c, k] + convB[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *f32, (B, S, H), contiguous
    convW_ptr,     # *f32, (H, H, 4), contiguous
    convB_ptr,     # *f32, (H,), contiguous
    out_ptr,       # *f32, (B, H, S), contiguous
    B: tl.int32, S: tl.int32, H: tl.int32,
    Bx_stride0, Bx_stride1, Bx_stride2,
    convW_stride0, convW_stride1, convW_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)          # batch
    c_id = tl.program_id(1)          # output channel
    s_block = tl.program_id(2)       # tile over S
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over the 4 taps
    for k in range(4):
        t_offsets = s_offsets + k - 1
        mask = (t_offsets >= 0) & (t_offsets < S) & s_mask
        # Load Bx[b, c, t_offsets]
        bx_ptrs = Bx_ptr + b_id * Bx_stride0 + c_id * Bx_stride1 + t_offsets * Bx_stride2
        bx_vals = tl.load(bx_ptrs, mask=mask, other=0.0)
        # Load convW[c, c, k] (scalar)
        w_val = tl.load(convW_ptr + c_id * convW_stride0 + c_id * convW_stride1 + k * convW_stride2)
        acc += bx_vals * w_val

    # Add bias
    bias_val = tl.load(convB_ptr + c_id)
    acc += bias_val

    # Store conv_out[b, c, s]
    out_ptrs = out_ptr + b_id * out_stride0 + c_id * out_stride1 + s_offsets * out_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 4) Final linear projection: y[B, S, H] = conv_out[B, H, S] @ out_proj_weight^T + out_proj_bias
#    We implement the outer-product accumulation over H in chunks.
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
    h_out = tl.program_id(1)  # output channel (H dimension)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # yin[b, k, s] for all s in tile and k chunk
        y_ptrs = yin_ptr + b_id * yin_stride0 + k_offsets[:, None] * yin_stride1 + s_offsets[None, :] * yin_stride2
        y_mask = k_mask[:, None] & s_mask[None, :]
        y_vals = tl.load(y_ptrs, mask=y_mask, other=0.0)  # [BLOCK_K, BLOCK_S]

        # Wout[h_out, k] for k chunk
        w_ptrs = Wout_ptr + h_out * Wout_stride0 + k_offsets * Wout_stride1
        w_vals = tl.load(w_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        acc += tl.sum(y_vals * w_vals[:, None], axis=0)

    # Add bias
    bias_val = tl.load(bout_ptr + h_out)
    acc += bias_val

    # Store yout[b, s, h_out]
    yout_ptrs = yout_ptr + b_id * yout_stride0 + s_offsets * yout_stride1 + h_out * yout_stride2
    tl.store(yout_ptrs, acc, mask=s_mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # x: (B, S, H), float32, contiguous
        B, S, H = x.shape
        device = x.device

        # 1) Triple linear projection: three outputs, each (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Grid: (B, H, tiles of S)
        BLOCK_S = 128
        grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        linear_bsh_kernel[grid](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        linear_bsh_kernel[grid](
            x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H], C_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        linear_bsh_kernel[grid](
            x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H], X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * X
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution: conv_out (B, H, S)
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y (B, H, S) -> out (B, S, H)
        out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_final = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        final_linear_bsh_kernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
