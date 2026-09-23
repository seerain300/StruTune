import torch
import triton
import triton.language as tl

# 1) Triple linear projection: out[B, S, H] = x @ weight^T + bias
#    where weight is (H, H) for each of the three outputs B, C, x_proj
@triton.jit
def linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    weight_ptr,    # *f32, (H, H)
    bias_ptr,      # *f32, (H,)
    out_ptr,       # *f32, (B, S, H)
    B, S, H,
    x_stride_b, x_stride_s, x_stride_h,
    weight_stride_0, weight_stride_1,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)   # batch index
    pid_h = tl.program_id(1)   # output feature index
    pid_s = tl.program_id(2)   # tile index along sequence

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # out[b, s, h] = sum_k x[b, s, k] * weight[h, k] + bias[h]
    for k in range(0, H):
        x_val = tl.load(x_ptr + pid_b * x_stride_b + s_offsets * x_stride_s + k * x_stride_h, mask=mask_s, other=0.0)
        w_val = tl.load(weight_ptr + pid_h * weight_stride_0 + k * weight_stride_1)
        acc += x_val * w_val

    # Add bias
    b_val = tl.load(bias_ptr + pid_h)
    acc += b_val

    # Store to out[b, s, h]
    tl.store(out_ptr + pid_b * out_stride_b + s_offsets * out_stride_s + pid_h * out_stride_h, acc, mask=mask_s)


# 2) Element-wise gating: out = A * B elementwise, A and B are (B, S, H)
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

    A_vals = tl.load(A_ptr + pid_b * A_stride_b + s_offsets * A_stride_s + pid_h * A_stride_h, mask=mask_s, other=0.0)
    B_vals = tl.load(B_ptr + pid_b * B_stride_b + s_offsets * B_stride_s + pid_h * B_stride_h, mask=mask_s, other=0.0)
    Out_vals = A_vals * B_vals

    tl.store(Out_ptr + pid_b * Out_stride_b + s_offsets * Out_stride_s + pid_h * Out_stride_h, Out_vals, mask=mask_s)


# 3) Grouped causal 1D convolution with kernel_size=4 and groups=H:
#    Input Bx has shape (B, S, H). We treat it as (B, N, H) where N=S.
#    conv_weight: (H, H, 4), conv_bias: (H,).
#    conv_out: (B, H, S) where conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t+k] * conv_weight[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,           # *f32, (B, S, H)
    convW_ptr,        # *f32, (H, H, 4)
    convB_ptr,        # *f32, (H,)
    conv_out_ptr,     # *f32, (B, H, S)
    B, S, H,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride_0, convW_stride_1, convW_stride_2,  # strides for (H, H, 4)
    conv_out_stride_b, conv_out_stride_h, conv_out_stride_s,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # output channel index
    pid_s = tl.program_id(2)  # tile along S

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # k in [0..3] for kernel_size=4 (causal: t+k)
    for k in range(4):
        t = s_offsets + k
        mask_t = (t < S) & mask_s
        vals = tl.load(Bx_ptr + pid_b * Bx_stride_b + t * Bx_stride_s + pid_c * Bx_stride_h, mask=mask_t, other=0.0)
        w_val = tl.load(convW_ptr + pid_c * convW_stride_0 + pid_c * convW_stride_1 + k * convW_stride_2)
        acc += vals * w_val

    # Add bias
    b_val = tl.load(convB_ptr + pid_c)
    acc += b_val

    # Store conv_out[b, c, s]
    tl.store(conv_out_ptr + pid_b * conv_out_stride_b + pid_c * conv_out_stride_h + s_offsets * conv_out_stride_s, acc, mask=mask_s)


# 4) Final linear projection: out[B,S,H] = y @ out_proj_weight^T + out_proj_bias
#    y is (B,H,S), out_proj_weight is (H,H), out_proj_bias is (H,)
@triton.jit
def final_linear_bsh_from_y_kernel(
    y_ptr,           # *f32, (B, H, S)
    weight_ptr,      # *f32, (H, H)
    bias_ptr,        # *f32, (H,)
    out_ptr,         # *f32, (B, S, H)
    B, S, H,
    y_stride_b, y_stride_h, y_stride_s,
    weight_stride_0, weight_stride_1,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)   # batch index
    pid_h = tl.program_id(1)   # output feature index
    pid_s = tl.program_id(2)   # tile along sequence

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # out[b, s, h] = sum_c y[b, c, s] * weight[h, c]
    for c in range(0, H):
        y_vals = tl.load(y_ptr + pid_b * y_stride_b + c * y_stride_h + s_offsets * y_stride_s, mask=mask_s, other=0.0)
        w_vals = tl.load(weight_ptr + pid_h * weight_stride_0 + c * weight_stride_1)  # scalar
        acc += y_vals * w_vals

    # Add bias
    b_val = tl.load(bias_ptr + pid_h)
    acc += b_val

    # Store to out[b, s, h]
    tl.store(out_ptr + pid_b * out_stride_b + s_offsets * out_stride_s + pid_h * out_stride_h, acc, mask=mask_s)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Shapes:
        # x: (B, S, H)
        # in_proj_weight: (3*H, H)
        # in_proj_bias: (3*H,)
        # conv_weight: (H, H, 4), conv_bias: (H,)
        # out_proj_weight: (H, H), out_proj_bias: (H,)

        B, S, H = x.shape
        device = x.device
        dtype = x.dtype

        # 1) Triple linear projection: slice in_proj_weight to (H,H) for each of three outputs
        W0 = in_proj_weight[:H, :]        # (H,H)
        b0 = in_proj_bias[:H]             # (H,)
        W1 = in_proj_weight[H:2*H, :]     # (H,H)
        b1 = in_proj_bias[2*H:3*H]        # (H,)
        W2 = in_proj_weight[2*H:3*H, :]   # (H,H)
        b2 = in_proj_bias[2*H:3*H]        # (H,)

        # Allocate outputs (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=dtype)
        C_out = torch.empty((B, S, H), device=device, dtype=dtype)
        X_out = torch.empty((B, S, H), device=device, dtype=dtype)

        # Launch triple linear kernel three times
        grid = (B, H, (S + 128 - 1) // 128)
        linear_bsh_kernel[grid](
            x, W0, b0, B_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        linear_bsh_kernel[grid](
            x, W1, b1, C_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        linear_bsh_kernel[grid](
            x, W2, b2, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
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

        # 3) Grouped causal 1D convolution: conv_out[b, c, t] with kernel_size=4, groups=H
        convW = conv_weight.contiguous()  # (H,H,4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)  # output shape (B,H,S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out (conv_out is (B,H,S), C_out is (B,H,S))
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

        # 5) Final linear projection: out[B,S,H] = y @ out_proj_weight^T + out_proj_bias
        out = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_final = (B, H, (S + 128 - 1) // 128)
        final_linear_bsh_from_y_kernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
