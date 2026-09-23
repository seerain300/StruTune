import torch
import triton
import triton.language as tl

# 1) Triple linear projection: compute out[B, S, H] = x @ weight^T + bias
#    where weight is (H, H) for each of the three outputs B, C, x_proj
@triton.jit
def linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    weight_ptr,    # *f32, (H, H)
    bias_ptr,      # *f32, (H,)
    out_ptr,       # *f32, (B, S, H)
    B, S, H,
    x_stride_b, x_stride_s, x_stride_h,
    weight_stride_0, weight_stride_1,  # (H,H)
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)  # output channel index in [0, H)
    pid_s = tl.program_id(2)  # tile over S

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Initialize accumulator for this (b, h) over tile of S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Sum over input dimension H (K = H): out[b, s, h] = sum_k x[b, s, k] * weight[h, k] + bias[h]
    for k in range(0, H):
        x_val = tl.load(x_ptr + pid_b * x_stride_b + s_offsets * x_stride_s + k * x_stride_h, mask=mask_s, other=0.0)
        w_val = tl.load(weight_ptr + pid_h * weight_stride_0 + k * weight_stride_1)  # (H,H)
        acc += x_val * w_val

    b_val = tl.load(bias_ptr + pid_h)
    acc += b_val

    # Store to out[b, s, h]
    tl.store(out_ptr + pid_b * out_stride_b + s_offsets * out_stride_s + pid_h * out_stride_h, acc, mask=mask_s)


# 2) Element-wise gating: Bx = B * X, elementwise, shape (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    A_ptr, B_ptr, Out_ptr,
    B, S, H,
    A_stride_b, A_stride_s, A_stride_h,
    B_stride_b, B_stride_s, B_stride_h,
    Out_stride_b, Out_stride_s, Out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    a = tl.load(A_ptr + pid_b * A_stride_b + s_offsets * A_stride_s + pid_h * A_stride_h, mask=mask_s, other=0.0)
    b = tl.load(B_ptr + pid_b * B_stride_b + s_offsets * B_stride_s + pid_h * B_stride_h, mask=mask_s, other=0.0)
    out = a * b
    tl.store(Out_ptr + pid_b * Out_stride_b + s_offsets * Out_stride_s + pid_h * Out_stride_h, out, mask=mask_s)


# 3) Grouped causal 1D convolution: conv_out[B, H, S] where groups=H, kernel_size=4
#    Input is Bx padded on left by 3: Bx_padded[B, S+3, H] = Bx[B, S, H] padded as [t-1, t, t+1, t+2]
#    conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t+k] * conv_weight[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,           # *f32, (B, S, H) - the original Bx; we will index as padded via offset
    convW_ptr,        # *f32, (H, H, 4) grouped by H
    convB_ptr,        # *f32, (H,)
    conv_out_ptr,     # *f32, (B, H, S)
    B, S, H,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride_0, convW_stride_1, convW_stride_2,  # strides for (H, H, 4)
    conv_out_stride_b, conv_out_stride_h, conv_out_stride_s,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # output channel
    pid_s = tl.program_id(2)  # tile over S

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # For kernel_size=4 causal conv (left padding 3), sum over k in [0..3]
    for k in range(4):
        t = s_offsets + k  # left-padding uses t+k, no subtraction
        mask_t = (t >= 0) & (t < S)
        vals = tl.load(Bx_ptr + pid_b * Bx_stride_b + t * Bx_stride_s + pid_c * Bx_stride_h, mask=mask_t & mask_s, other=0.0)
        w_val = tl.load(convW_ptr + pid_c * convW_stride_0 + pid_c * convW_stride_1 + k * convW_stride_2)
        acc += vals * w_val

    # Add bias
    b_val = tl.load(convB_ptr + pid_c)
    acc += b_val

    # Store to conv_out[b, c, s]
    tl.store(conv_out_ptr + pid_b * conv_out_stride_b + pid_c * conv_out_stride_h + s_offsets * conv_out_stride_s, acc, mask=mask_s)


# 4) Final linear projection: out[B,S,H] = y @ out_proj_weight^T + out_proj_bias
#    where y is (B,S,H)
@triton.jit
def final_linear_bsh_from_y_kernel(
    y_ptr,            # *f32, (B, S, H)
    out_proj_ptr,     # *f32, (H, H)
    out_proj_bias_ptr,  # *f32, (H,)
    out_ptr,          # *f32, (B, S, H)
    B, S, H,
    y_stride_b, y_stride_s, y_stride_h,
    out_proj_stride_0, out_proj_stride_1,  # (H,H)
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)  # output channel index
    pid_s = tl.program_id(2)  # tile over S

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for k in range(0, H):  # input reduction over H
        y_val = tl.load(y_ptr + pid_b * y_stride_b + s_offsets * y_stride_s + k * y_stride_h, mask=mask_s, other=0.0)
        w_val = tl.load(out_proj_ptr + pid_h * out_proj_stride_0 + k * out_proj_stride_1)
        acc += y_val * w_val

    b_val = tl.load(out_proj_bias_ptr + pid_h)
    acc += b_val

    tl.store(out_ptr + pid_b * out_stride_b + s_offsets * out_stride_s + pid_h * out_stride_h, acc, mask=mask_s)

# ----------------------------
# Host-side forward (ModelNew)
# ----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 for consistent computation
        device = x.device
        dtype = torch.float32
        B, S, H = x.shape

        # 1) Triple linear projection: compute B, C, X (B,S,H) from x
        # Slice in_proj_weight into (H,H) for each of the three outputs
        W0 = in_proj_weight[:H, :].contiguous()  # (H,H)
        b0 = in_proj_bias[:H].contiguous()       # (H,)
        B_out = torch.empty((B, S, H), device=device, dtype=dtype)
        linear_bsh_kernel[(B, H, (S + 128 - 1) // 128)](
            x, W0, b0, B_out, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        W1 = in_proj_weight[H:2*H, :].contiguous()  # (H,H)
        b1 = in_proj_bias[H:2*H].contiguous()       # (H,)
        C_out = torch.empty((B, S, H), device=device, dtype=dtype)
        linear_bsh_kernel[(B, H, (S + 128 - 1) // 128)](
            x, W1, b1, C_out, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        W2 = in_proj_weight[2*H:3*H, :].contiguous()  # (H,H)
        b2 = in_proj_bias[2*H:3*H].contiguous()       # (H,)
        X_out = torch.empty((B, S, H), device=device, dtype=dtype)
        linear_bsh_kernel[(B, H, (S + 128 - 1) // 128)](
            x, W2, b2, X_out, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * X
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)
        elemwise_mul_bsh_kernel[(B, H, (S + 128 - 1) // 128)](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D convolution on Bx, groups=H, kernel_size=4
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)  # output is (B,H,S)
        grouped_causal_conv1d_kernel[(B, H, (S + 128 - 1) // 128)](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out, conv_out is (B,H,S); transpose to (B,S,H)
        conv_out_t = conv_out.transpose(1, 2).contiguous()  # (B,S,H)
        y = torch.empty((B, S, H), device=device, dtype=dtype)
        elemwise_mul_bsh_kernel[(B, H, (S + 128 - 1) // 128)](
            C_out, conv_out_t, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out_t.stride(0), conv_out_t.stride(1), conv_out_t.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 5) Final linear projection to output (B,S,H)
        out = torch.empty((B, S, H), device=device, dtype=dtype)
        out_proj = out_proj_weight.contiguous()            # (H,H)
        out_proj_bias = out_proj_bias.contiguous()         # (H,)
        final_linear_bsh_from_y_kernel[(B, H, (S + 128 - 1) // 128)](
            y, out_proj, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj.stride(0), out_proj.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
