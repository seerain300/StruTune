import torch
import triton
import triton.language as tl

# 1) Triple linear projection kernel: out[B, S, H] = x @ W^T + b
#    We launch this kernel three times to produce B, C, x_proj.
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,      # *f32, (B, S, H)
    W0_ptr, b0_ptr,  # *f32, (H, H), (H,)
    W1_ptr, b1_ptr,  # *f32, (H, H), (H,)
    W2_ptr, b2_ptr,  # *f32, (H, H), (H,)
    B_out_ptr, C_out_ptr, X_out_ptr,  # *f32, (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    x_stride0, x_stride1, x_stride2,
    W0_stride0, W0_stride1,
    W1_stride0, W1_stride1,
    W2_stride0, W2_stride1,
    B_stride0, B_stride1, B_stride2,
    C_stride0, C_stride1, C_stride2,
    X_stride0, X_stride1, X_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (B, H, tiles of S)
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Accumulate over K=H in chunks
    for k0 in range(0, H, 64):  # 64 works well for small H; masked for any H
        k_offsets = k0 + tl.arange(0, 64)
        k_mask = k_offsets < H

        # Load x[b, s, k]
        x_ptrs = x_ptr + b_id * x_stride0 + s_offsets[None, :] * x_stride1 + k_offsets[:, None] * x_stride2
        x_mask = s_mask[None, :] & k_mask[:, None]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [64, BLOCK_S]

        # Load weights W[h, k]
        # For each weight set (W0, W1, W2), accumulate into acc
        # W0
        w0_ptrs = W0_ptr + h_id * W0_stride0 + k_offsets * W0_stride1
        w0 = tl.load(w0_ptrs, mask=k_mask, other=0.0)  # [64]
        acc += tl.sum(x_vals * w0[None, :], axis=0)

        # W1 (for C_out)
        w1_ptrs = W1_ptr + h_id * W1_stride0 + k_offsets * W1_stride1
        w1 = tl.load(w1_ptrs, mask=k_mask, other=0.0)  # [64]
        # We need to store to C_out here as well
        c_ptrs = C_out_ptr + b_id * C_stride0 + h_id * C_stride1 + s_offsets * C_stride2
        # Compute B_out acc for this h_id and store
        b_ptrs = B_out_ptr + b_id * B_stride0 + h_id * B_stride1 + s_offsets * B_stride2
        # Add bias for B_out
        b0 = tl.load(b0_ptr + h_id, mask=True, other=0.0).to(tl.float32)
        tl.store(b_ptrs, acc + b0, mask=s_mask)

        # Store C_out = acc + b1 (need to add b1 now)
        c_vals = acc + tl.load(b1_ptr + h_id, mask=True, other=0.0).to(tl.float32)
        tl.store(c_ptrs, c_vals, mask=s_mask)

        # W2 for X_out
        w2_ptrs = W2_ptr + h_id * W2_stride0 + k_offsets * W2_stride1
        w2 = tl.load(w2_ptrs, mask=k_mask, other=0.0)  # [64]
        x_ptrs_out = X_out_ptr + b_id * X_stride0 + h_id * X_stride1 + s_offsets * X_stride2
        x_vals_acc = acc + tl.load(b2_ptr + h_id, mask=True, other=0.0).to(tl.float32)
        tl.store(x_ptrs_out, x_vals_acc, mask=s_mask)

# 2) Element-wise gating: Bx = B * X (elementwise over (B, H, S))
@triton.jit
def elemwise_mul_bsh_kernel(
    B_ptr, X_ptr, Out_ptr,
    B: tl.int32, S: tl.int32, H: tl.int32,
    B_stride0, B_stride1, B_stride2,
    X_stride0, X_stride1, X_stride2,
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

    b_ptrs = B_ptr + b_id * B_stride0 + h_id * B_stride1 + s_offsets * B_stride2
    x_ptrs = X_ptr + b_id * X_stride0 + h_id * X_stride1 + s_offsets * X_stride2
    out_ptrs = Out_ptr + b_id * Out_stride0 + h_id * Out_stride1 + s_offsets * Out_stride2

    b_vals = tl.load(b_ptrs, mask=s_mask, other=0.0)
    x_vals = tl.load(x_ptrs, mask=s_mask, other=0.0)
    out_vals = b_vals * x_vals
    tl.store(out_ptrs, out_vals, mask=s_mask)

# 3) Grouped causal 1D convolution: conv_out[B, H, S] from Bx[B, H, S] and conv_weight[H, H, 4], bias[H], groups=H
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convB_ptr, convOut_ptr,
    B: tl.int32, S: tl.int32, H: tl.int32,
    Bx_stride0, Bx_stride1, Bx_stride2,
    convW_stride0, convW_stride1, convW_stride2,
    convOut_stride0, convOut_stride1, convOut_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # For groups=H, conv_out[b, h, s] = sum_{k=0..3} Bx[b, h, s+k-1] * convW[h, h, k] + convB[h]
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # k in [0..3]
    for k in range(4):
        t_offsets = s_offsets + k - 1  # causal left pad
        t_mask = (t_offsets >= 0) & (t_offsets < S) & s_mask
        bx_ptrs = Bx_ptr + b_id * Bx_stride0 + h_id * Bx_stride1 + t_offsets * Bx_stride2
        bx_vals = tl.load(bx_ptrs, mask=t_mask, other=0.0)  # [BLOCK_S]

        w_ptr = convW_ptr + h_id * convW_stride0 + h_id * convW_stride1 + k * convW_stride2
        w_val = tl.load(w_ptr)  # scalar
        acc += bx_vals * w_val

    bias_val = tl.load(convB_ptr + h_id)
    acc += bias_val
    out_ptrs = convOut_ptr + b_id * convOut_stride0 + h_id * convOut_stride1 + s_offsets * convOut_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 4) Output gating: y[B, H, S] = C[B, H, S] * convOut[B, H, S]
@triton.jit
def elemwise_mul_bhs_kernel(
    C_ptr, convOut_ptr, y_ptr,
    B: tl.int32, S: tl.int32, H: tl.int32,
    C_stride0, C_stride1, C_stride2,
    convOut_stride0, convOut_stride1, convOut_stride2,
    y_stride0, y_stride1, y_stride2,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (B, H, tiles of S)
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    c_ptrs = C_ptr + b_id * C_stride0 + h_id * C_stride1 + s_offsets * C_stride2
    co_ptrs = convOut_ptr + b_id * convOut_stride0 + h_id * convOut_stride1 + s_offsets * convOut_stride2
    y_ptrs = y_ptr + b_id * y_stride0 + h_id * y_stride1 + s_offsets * y_stride2

    c_vals = tl.load(c_ptrs, mask=s_mask, other=0.0)
    co_vals = tl.load(co_ptrs, mask=s_mask, other=0.0)
    y_vals = c_vals * co_vals
    tl.store(y_ptrs, y_vals, mask=s_mask)

# 5) Final linear projection: y[B, S, H] = y[B, H, S] @ out_proj_weight^T + out_proj_bias
@triton.jit
def final_linear_bsh_kernel(
    yin_ptr,    # *f32, (B, H, S), contiguous
    Wout_ptr,   # *f32, (H, H), contiguous
    bout_ptr,   # *f32, (H,), contiguous
    yout_ptr,   # *f32, (B, S, H), contiguous
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

        # yin[b, k, s]
        y_ptrs = yin_ptr + b_id * yin_stride0 + k_offsets[:, None] * yin_stride1 + s_offsets[None, :] * yin_stride2
        y_mask = k_mask[:, None] & s_mask[None, :]
        y_vals = tl.load(y_ptrs, mask=y_mask, other=0.0)  # [BLOCK_K, BLOCK_S]

        # Wout[h_out, k]
        w_ptrs = Wout_ptr + h_out * Wout_stride0 + k_offsets * Wout_stride1
        w_vals = tl.load(w_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Accumulate outer products over k chunk
        # y_vals[k, s] * w_vals[k] -> reduce over k
        # Triton does elementwise multiply, so we need a reduction over axis=0:
        acc += tl.sum(y_vals * w_vals[None, :], axis=0)

    # Add bias
    b_out = tl.load(bout_ptr + h_out)
    acc += b_out

    # Store yout[b, s, h_out]
    yout_ptrs = yout_ptr + b_id * yout_stride0 + s_offsets * yout_stride2 + h_out * yout_stride1
    tl.store(yout_ptrs, acc, mask=s_mask)

# Host function ModelNew.forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized forward that mirrors the original computation:
        1) Triple linear projection
        2) Element-wise gating: Bx = B * x_proj
        3) Grouped causal conv1d on Bx with kernel_size=4, groups=hidden_size
        4) Output gating: y = C * conv_out
        5) Final output projection
        """
        B, S, H = x.shape
        device = x.device
        dtype = torch.float32  # Ensure float32 for correctness

        # 1) Triple linear projection
        # Slice in_proj_weight into three groups: (H,H)
        W0 = in_proj_weight[:, :H]       # (H,H)
        b0 = in_proj_bias[:H]            # (H,)
        W1 = in_proj_weight[:, H:2*H]    # (H,H)
        b1 = in_proj_bias[2*H:3*H]       # (H,)
        W2 = in_proj_weight[:, 2*H:3*H]  # (H,H)
        b2 = in_proj_bias[3*H:4*H]       # (H,)

        # Allocate outputs
        B_out = torch.empty((B, S, H), device=device, dtype=dtype)
        C_out = torch.empty((B, S, H), device=device, dtype=dtype)
        X_out = torch.empty((B, S, H), device=device, dtype=dtype)

        # Kernel launch grid
        grid = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid](
            x, W0, b0, W1, b1, W2, b2, B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 3) Grouped causal conv1d: conv_out (B,H,S)
        convW = conv_weight.contiguous()  # (H,H,4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        grouped_causal_conv1d_kernel[(B, H, (S + 128 - 1) // 128)](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> (B,H,S)
        y = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bhs_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y -> output (B,S,H)
        # y shape (B,H,S), out_proj_weight (H,H), out_proj_bias (H)
        yT = y.transpose(-1, -2).contiguous()  # (B,S,H) for kernel expectation
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        final_grid = (B, H, (S + 128 - 1) // 128)
        final_linear_bsh_kernel[final_grid](
            yT, out_proj_weight, out_proj_bias, output,
            B, S, H,
            yT.stride(0), yT.stride(1), yT.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
