import torch
import triton
import triton.language as tl

# 1) Triple linear projection: out[B, S, H] = x @ W^T + b
@triton.jit
def linear_bsh_kernel(
    x_ptr,      # *f32, (B, S, H)
    W_ptr,      # *f32, (H, H)
    b_ptr,      # *f32, (H,)
    out_ptr,    # *f32, (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    x_stride0, x_stride1, x_stride2,
    W_stride0, W_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)  # batch
    h_out = tl.program_id(1)  # output channel index in [0..H)
    s_block = tl.program_id(2)
    s_start = s_block * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Iterate over K=H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load x[b, s, k] for all s in tile and k chunk: x shape (B, S, H)
        x_ptrs = x_ptr + b_id * x_stride0 + s_offsets[None, :] * x_stride1 + k_offsets[:, None] * x_stride2
        x_mask = s_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_S, BLOCK_K]

        # Load W[h_out, k] for k chunk: W shape (H, H)
        W_ptrs = W_ptr + h_out * W_stride0 + k_offsets * W_stride1
        W_vals = tl.load(W_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Accumulate: acc += sum_k x_vals[:, k] * W_vals[k]
        # x_vals: [BLOCK_S, BLOCK_K], W_vals: [BLOCK_K] -> broadcast to [BLOCK_S, BLOCK_K], then sum over axis=1
        acc += tl.sum(x_vals * W_vals[None, :], axis=1)

    # Add bias[h_out]
    bias_val = tl.load(b_ptr + h_out)
    acc += bias_val

    # Store to out[b, h_out, s]
    out_ptrs = out_ptr + b_id * out_stride0 + h_out * out_stride1 + s_offsets * out_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 2) Element-wise gating: Bx = B * x_proj
@triton.jit
def elemwise_mul_bhs_kernel(
    B_ptr, X_ptr, out_ptr,  # *f32
    B: tl.int32, S: tl.int32, H: tl.int32,
    B_stride0, B_stride1, B_stride2,
    X_stride0, X_stride1, X_stride2,
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

    B_ptrs = B_ptr + b_id * B_stride0 + h_id * B_stride1 + s_offsets * B_stride2
    X_ptrs = X_ptr + b_id * X_stride0 + h_id * X_stride1 + s_offsets * X_stride2
    out_ptrs = out_ptr + b_id * out_stride0 + h_id * out_stride1 + s_offsets * out_stride2

    b_vals = tl.load(B_ptrs, mask=s_mask, other=0.0)
    x_vals = tl.load(X_ptrs, mask=s_mask, other=0.0)
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

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # kernel_size = 4: sum over k=0..3, with left pad = 3 (causal)
    for k in range(4):
        t_offsets = s_offsets + k - 1  # causal left shift
        t_mask = (t_offsets >= 0) & (t_offsets < S) & s_mask
        bx_ptrs = Bx_ptr + b_id * Bx_stride0 + h_id * Bx_stride1 + t_offsets * Bx_stride2
        bx_vals = tl.load(bx_ptrs, mask=t_mask, other=0.0)  # [BLOCK_S]
        # conv_weight indexing: (H, H, 4), groups=H -> we only use convW[h_id, h_id, k]
        w_ptr = convW_ptr + h_id * convW_stride0 + h_id * convW_stride1 + k * convW_stride2
        w_val = tl.load(w_ptr)  # scalar
        acc += bx_vals * w_val

    # Add bias
    bias_val = tl.load(convB_ptr + h_id)
    acc += bias_val

    # Store conv_out[b, h_id, s]
    out_ptrs = convOut_ptr + b_id * convOut_stride0 + h_id * convOut_stride1 + s_offsets * convOut_stride2
    tl.store(out_ptrs, acc, mask=s_mask)

# 4) Final linear projection: out[B, S, H] = conv_out[B, H, S] @ out_proj_weight^T + out_proj_bias
@triton.jit
def final_linear_bsh_kernel(
    yin_ptr,     # *f32, (B, H, S)
    Wout_ptr,    # *f32, (H, H)
    bout_ptr,    # *f32, (H,)
    yout_ptr,    # *f32, (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    yin_stride0, yin_stride1, yin_stride2,
    Wout_stride0, Wout_stride1,
    yout_stride0, yout_stride1, yout_stride2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b_id = tl.program_id(0)  # batch
    s_block = tl.program_id(1)
    h_out = tl.program_id(2)  # output channel index
    s_start = s_block * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Iterate over K=H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load yin[b, k, s] for all s in tile and k chunk: yin shape (B, H, S)
        y_ptrs = yin_ptr + b_id * yin_stride0 + k_offsets[:, None] * yin_stride1 + s_offsets[None, :] * yin_stride2
        y_mask = k_mask[:, None] & s_mask[None, :]
        y_vals = tl.load(y_ptrs, mask=y_mask, other=0.0)  # [BLOCK_K, BLOCK_S]

        # Load Wout[h_out, k] for k chunk
        W_ptrs = Wout_ptr + h_out * Wout_stride0 + k_offsets * Wout_stride1
        W_vals = tl.load(W_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Accumulate: acc += sum_k y_vals[k, :] * W_vals[k]
        acc += tl.sum(y_vals * W_vals[None, :], axis=0)

    # Add bias[h_out]
    bias_val = tl.load(bout_ptr + h_out)
    acc += bias_val

    # Store to yout[b, s, h_out]
    yout_ptrs = yout_ptr + b_id * yout_stride0 + s_offsets * yout_stride1 + h_out * yout_stride2
    tl.store(yout_ptrs, acc, mask=s_mask)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized fused computation matching the original:
        1) Triple linear projection
        2) Element-wise gating
        3) Grouped causal 1D conv (kernel_size=4, groups=H)
        4) Output gating
        5) Final output projection
        """
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda \
            and conv_weight.is_cuda and conv_bias.is_cuda \
            and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be on CUDA for Triton."

        B, S, H = x.shape

        # 1) Triple linear projection into B, C, x_proj, each (B, S, H)
        x_c = x.contiguous()
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # W0, b0: first H rows of in_proj_weight and bias
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()
        # W1, b1: middle H
        W1 = in_proj_weight[H:2*H, :].contiguous()
        b1 = in_proj_bias[H:2*H].contiguous()
        # W2, b2: last H
        W2 = in_proj_weight[2*H:3*H, :].contiguous()
        b2 = in_proj_bias[2*H:3*H].contiguous()

        BLOCK_S = 128
        grid_linear = (B, H, triton.cdiv(S, BLOCK_S))
        linear_bsh_kernel[grid_linear](
            x_c, W0, b0, B_out,
            B, S, H,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )
        linear_bsh_kernel[grid_linear](
            x_c, W1, b1, C_out,
            B, S, H,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )
        linear_bsh_kernel[grid_linear](
            x_c, W2, b2, X_out,
            B, S, H,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            W2.stride(0), W2.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * X
        Bx = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid_mul = (B, H, triton.cdiv(S, BLOCK_S))
        elemwise_mul_bhs_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 3) Grouped causal conv1d: conv_out[B, H, S]
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid_conv = (B, H, triton.cdiv(S, BLOCK_S))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C * conv_out -> (B, H, S)
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid_mul2 = (B, H, triton.cdiv(S, BLOCK_S))
        elemwise_mul_bhs_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y -> out (B, S, H)
        y_c = y.contiguous()  # (B, H, S)
        out_proj_weight_c = out_proj_weight.contiguous()  # (H, H)
        out_proj_bias_c = out_proj_bias.contiguous()      # (H,)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid_final = (B, triton.cdiv(S, BLOCK_S), H)
        final_linear_bsh_kernel[grid_final](
            y_c, out_proj_weight_c, out_proj_bias_c, output,
            B, S, H,
            y_c.stride(0), y_c.stride(1), y_c.stride(2),
            out_proj_weight_c.stride(0), out_proj_weight_c.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_K=64, num_warps=4, num_stages=2
        )

        return output

# The original helper functions from the prompt can be reused for testing:
@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    # Original computation for reference
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]
    BCx = F.linear(x, in_proj_weight, in_proj_bias).transpose(-1, -2)  # (B, 3H, S)
    B, C, x_proj = BCx.chunk(3, dim=1)  # each (B, H, S)
    Bx = B * x_proj
    Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))  # causal pad
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)  # (B, H, S)
    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()  # (B, S, H)
    output = F.linear(y, out_proj_weight, out_proj_bias)
    return output

class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)

# Example of how to use ModelNew:
# model = ModelNew().cuda()
# x = torch.randn(2, 4096, 128, device='cuda', dtype=torch.float32)
# in_proj_weight = torch.randn(384, 128, device='cuda', dtype=torch.float32)  # 3*128
# in_proj_bias = torch.randn(384, device='cuda', dtype=torch.float32)
# conv_weight = torch.randn(128, 128, 4, device='cuda', dtype=torch.float32)  # (H, H, 4)
# conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
# out_proj_weight = torch.randn(128, 128, device='cuda', dtype=torch.float32)
# out_proj_bias = torch.randn(128, device='cuda', dtype=torch.float32)
# y = model(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)
# For correctness check, compare with original Model().cuda()(x, ...).


def run(*args):
    return ModelNew()(*args)
