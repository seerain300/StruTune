import torch
import triton
import triton.language as tl


# 1) Triple linear projection kernel: computes out[B, S, H] = x @ W^T + b
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    W_ptr, b_ptr,  # *f32, (H, H), (H,)
    out_ptr,       # *f32, (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    x_s0, x_s1, x_s2,
    W_s0, W_s1,
    out_s0, out_s1, out_s2,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for out[b, s_offsets, h]
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # x[b, s, k] vectorized over s_offsets
        x_ptrs = x_ptr + b * x_s0 + s_offsets[:, None] * x_s1 + k_offsets[None, :] * x_s2
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)

        # W[h, k]
        W_ptrs = W_ptr + h * W_s0 + k_offsets * W_s1
        W_vals = tl.load(W_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Accumulate: sum over k of x_vals[:, k] * W_vals[k]
        acc += tl.sum(x_vals * W_vals[None, :], axis=1)

    # Add bias[h]
    b_val = tl.load(b_ptr + h)
    acc += b_val

    # Store result to out[b, s_offsets, h]
    out_ptrs = out_ptr + b * out_s0 + s_offsets * out_s1 + h * out_s2
    tl.store(out_ptrs, acc, mask=mask_s)


# 2) Element-wise gating: out = a * b, both (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.int32, S: tl.int32, H: tl.int32,
    a_s0, a_s1, a_s2,
    b_s0, b_s1, b_s2,
    out_s0, out_s1, out_s2,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    a_ptrs = a_ptr + b * a_s0 + s_offsets * a_s1 + h * a_s2
    b_ptrs = b_ptr + b * b_s0 + s_offsets * b_s1 + h * b_s2
    out_ptrs = out_ptr + b * out_s0 + s_offsets * out_s1 + h * out_s2

    a_vals = tl.load(a_ptrs, mask=mask_s, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask_s, other=0.0)
    tl.store(out_ptrs, a_vals * b_vals, mask=mask_s)


# 3) Grouped causal 1D convolution: conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t+k-1] * convW[c, c, k] + convB[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,          # *f32, padded (B, H, S+pad), contiguous
    convW_ptr, convB_ptr,  # *f32, (H, H, 4), (H,)
    conv_out_ptr,    # *f32, (B, H, S)
    B: tl.int32, S: tl.int32, H: tl.int32,
    Bx_s0, Bx_s1, Bx_s2,
    convW_s0, convW_s1, convW_s2,
    conv_out_s0, conv_out_s1, conv_out_s2,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    tile_s = tl.program_id(2)

    t_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_t = t_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over k in [0,1,2,3]
    for k in range(4):
        t_src = t_offsets + k - 1  # causal: shifted by k-1
        # We rely on padding: Bx_padded has length S+pad. For t_src < S+pad, all are valid; mask_t ensures t_offsets < S.
        bx_ptrs = Bx_ptr + b * Bx_s0 + c * Bx_s1 + t_src * Bx_s2
        bx_vals = tl.load(bx_ptrs, mask=mask_t, other=0.0)

        # conv weight for channel c: convW[c, c, k]
        w_ptr = convW_ptr + c * convW_s0 + c * convW_s1 + k * convW_s2
        w_val = tl.load(w_ptr)
        acc += bx_vals * w_val

    # Add bias[c]
    b_val = tl.load(convB_ptr + c)
    acc += b_val

    # Store conv_out[b, c, t_offsets]
    out_ptrs = conv_out_ptr + b * conv_out_s0 + c * conv_out_s1 + t_offsets * conv_out_s2
    tl.store(out_ptrs, acc, mask=mask_t)


# 4) Output gating: y = C_out * conv_out, shape (B, H, S)
#   Same as elemwise_mul_bsh_kernel but inputs are (B, S, H) and we index as (b, s, h) for loads/stores.


# 5) Final linear projection: out[B, S, H] = y @ out_proj_weight^T + out_proj_bias
@triton.jit
def final_linear_bsh_kernel(
    y_ptr,           # *f32, (B, S, H)
    outW_ptr, outB_ptr,   # *f32, (H, H), (H,)
    out_ptr,         # *f32, (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    y_s0, y_s1, y_s2,
    outW_s0, outW_s1,
    out_s0, out_s1, out_s2,
    BLOCK_S: tl.constexpr, BLOCK_J: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for j_start in range(0, H, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        mask_j = j_offsets < H

        # y[b, s_offsets, j_offsets] -> shape [BLOCK_S, BLOCK_J]
        y_ptrs = y_ptr + b * y_s0 + s_offsets[:, None] * y_s1 + j_offsets[None, :] * y_s2
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None] & mask_j[None, :], other=0.0)

        # outW[h, j]
        outW_ptrs = outW_ptr + h * outW_s0 + j_offsets * outW_s1
        outW_vals = tl.load(outW_ptrs, mask=mask_j, other=0.0)  # [BLOCK_J]

        # Accumulate over J: sum_j y_vals[:, j] * outW_vals[j]
        acc += tl.sum(y_vals * outW_vals[None, :], axis=1)

    # Add bias[h]
    b_val = tl.load(outB_ptr + h)
    acc += b_val

    # Store to out[b, s_offsets, h]
    out_ptrs = out_ptr + b * out_s0 + s_offsets * out_s1 + h * out_s2
    tl.store(out_ptrs, acc, mask=mask_s)


def _triton_launch_triple_linear(x, W, b, out):
    B, S, H = x.shape
    grid = (B, H, (S + 128 - 1) // 128)
    triple_linear_bsh_kernel[grid](
        x, W, b, out,
        B, S, H,
        x.stride(0), x.stride(1), x.stride(2),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_S=128, BLOCK_K=64, num_warps=4, num_stages=2
    )


def _triton_elemwise_mul(a, b, out):
    B, S, H = a.shape
    grid = (B, H, (S + 128 - 1) // 128)
    elemwise_mul_bsh_kernel[grid](
        a, b, out,
        B, S, H,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_S=128, num_warps=4, num_stages=2
    )


def _triton_grouped_causal_conv(Bx, convW, convB, conv_out):
    B, S, H = Bx.shape
    pad = convW.shape[2] - 1  # kernel_size - 1
    Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # pad only on the right of S (causal left pad is handled in kernel)
    grid = (B, H, (S + 128 - 1) // 128)
    grouped_causal_conv1d_kernel[grid](
        Bx_padded, convW, convB, conv_out,
        B, S, H,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        convW.stride(0), convW.stride(1), convW.stride(2),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128, num_warps=4, num_stages=2
    )


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure float32
        x = x.float()
        in_proj_weight = in_proj_weight.float()
        in_proj_bias = in_proj_bias.float()
        conv_weight = conv_weight.float()
        conv_bias = conv_bias.float()
        out_proj_weight = out_proj_weight.float()
        out_proj_bias = out_proj_bias.float()

        B, S, H = x.shape

        # 1) Triple linear projection
        W0 = in_proj_weight[:H, :]     # (H, H)
        b0 = in_proj_bias[:H]          # (H,)
        W1 = in_proj_weight[H:2*H, :]  # (H, H)
        b1 = in_proj_bias[2*H:3*H]     # (H,)
        W2 = in_proj_weight[2*H:3*H, :]  # (H, H)
        b2 = in_proj_bias[3*H:4*H]     # (H,)

        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _triton_launch_triple_linear(x, W0, b0, B_out)

        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _triton_launch_triple_linear(x, W1, b1, C_out)

        X_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _triton_launch_triple_linear(x, W2, b2, X_out)

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _triton_elemwise_mul(B_out, X_out, Bx)

        # 3) Grouped causal 1D convolution: groups=H, kernel_size=conv_weight.shape[2] (which we assume is 4)
        # convW shape: (H, H, K) with K = conv_weight.shape[2]
        K = conv_weight.shape[2]
        convW = conv_weight
        convB = conv_bias
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        _triton_grouped_causal_conv(Bx, convW, convB, conv_out)

        # 4) Output gating: y = C_out * conv_out -> shape (B, H, S)
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        _triton_elemwise_mul(C_out, conv_out, y)  # y is (B, H, S)

        # 5) Final linear projection to (B, S, H): out = y @ out_proj_weight^T + out_proj_bias
        outB = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _triton_launch_triple_linear(y.transpose(-1, -2).contiguous(), out_proj_weight, out_proj_bias, outB)

        return outB

# The original run function expects a specific interface; ModelNew.forward adheres to the same signature.
# If you need to call it as ModelNew().forward(...), it will work as defined.


def run(*args):
    return ModelNew()(*args)
