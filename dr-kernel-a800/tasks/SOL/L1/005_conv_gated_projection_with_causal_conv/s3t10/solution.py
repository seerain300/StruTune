import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), where Nproj=3*H (but we don't use H; Nproj is provided)
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, shape (B, S, H)
    in_proj_weight_ptr,     # *f32, shape (Nproj, H)
    in_proj_bias_ptr,       # *f32, shape (Nproj,)
    BCx_ptr,                # *f32, shape (B, S, Nproj)
    B, S,                   # ints (runtime)
    H: tl.constexpr,        # int (compile-time constant for loop)
    Nproj: tl.constexpr,    # int (compile-time constant for output channels)
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    acc = 0.0
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating and split from BCx (conceptually transpose by channels)
# Read B (co=0) and x_proj (co=1) from BCx, compute Bx = B * x_proj
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, shape (B, S, Nproj) with Nproj=3*H
    Bx_ptr,         # *f32, shape (B, S, H)
    B, S,           # ints (runtime)
    H: tl.constexpr,         # int (compile-time constant for H)
    Nproj: tl.constexpr,     # int (compile-time constant for output channels)
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output index in [0, H)

    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx is logically indexed as (b, ci, t); we use (B, S, H) mapping (b, t, ci).
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, S, H) (we store per (b, t, ci))
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, S, H)  # we'll store per (b, t, ci)
    B, S, H,            # ints (runtime)
    stride_bx_b, stride_bx_s, stride_bx_h,  # strides for Bx: (b, s, h)
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides: (go, gi, k)
    stride_out_b, stride_out_s, stride_out_h,  # strides for conv_out: (b, s, h)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    t = tl.program_id(2)   # time index
    if b >= B or ci >= H or t >= S:
        return

    acc = 0.0
    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for k in range(0, 4):
        x_pos = t + k
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + t * stride_out_s + ci * stride_out_h, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, Nproj) with Nproj=3*H
    conv_out_ptr,   # *f32, shape (B, S, H)
    y_ptr,          # *f32, shape (B, S, H)
    B, S,           # ints (runtime)
    H: tl.constexpr,         # int (compile-time constant for H)
    Nproj: tl.constexpr,     # int (compile-time constant for output channels)
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_out_b, stride_out_s, stride_out_h,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h)
    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                 # *f32, shape (B, S, H)
    out_proj_weight_ptr,   # *f32, shape (H, H)
    out_proj_bias_ptr,     # *f32, shape (H,)
    output_ptr,            # *f32, shape (B, S, H)
    B, S,                  # ints (runtime)
    H: tl.constexpr,       # int (compile-time constant for H)
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)  # output channel index

    acc = 0.0
    for i in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + i * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_o + i * stride_w_i)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + h_out)
    acc += bias
    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure all tensors are contiguous and float32 for Triton kernels
        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        B, S, H = x.shape
        Nproj = 3 * H  # because in_proj_weight has 3 channels of size H

        # 1) Triple linear projection: BCx shape (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), device=x.device, dtype=torch.float32)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S,
            H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Element-wise gating: compute Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S,
            H, Nproj,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D convolution on Bx (kernel_size=4), groups=H
        conv_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)  # per (b, t, ci)
        grid3 = (B, H, S)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Output gating: y = C * conv_out
        y = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S,
            H, Nproj,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final output projection
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y, out_proj_weight, out_proj_bias, output,
            B, S,
            H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
