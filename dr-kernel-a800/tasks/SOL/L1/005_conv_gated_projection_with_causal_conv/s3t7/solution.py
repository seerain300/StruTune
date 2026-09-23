import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj)
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, shape (B, S, H)
    in_proj_weight_ptr,     # *f32, shape (Nproj, H)
    in_proj_bias_ptr,       # *f32, shape (Nproj,)
    BCx_ptr,                # *f32, shape (B, S, Nproj)
    B, S, H, Nproj,         # ints
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
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
    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating: Bx = B * x_proj
# BCx_T: (B, 3H, S) conceptual channels [0=B, 1=x_proj, 2=C]
# Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_T_ptr,              # *f32, shape (B, 3H, S)
    Bx_ptr,                 # *f32, shape (B, S, H)
    B, S, H, Nproj,         # ints
    stride_bcx_b, stride_bcx_ch, stride_bcx_s,
    stride_bx_b, stride_bx_s, stride_bx_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Read B from channel 0 and x_proj from channel 1
    B_val = tl.load(BCx_T_ptr + b * stride_bcx_b + 0 * stride_bcx_ch + s * stride_bcx_s)
    x_val = tl.load(BCx_T_ptr + b * stride_bcx_b + 1 * stride_bcx_ch + s * stride_bcx_s)

    bx = B_val * x_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: (B, H, S) conceptual indexing as (b, ci, t), where Bx is actually (B, S, H) but we map as (b, ci, t)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_ci, stride_bx_t,  # mapping for (b, ci, t) on Bx
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            # Load x[b, ci, x_pos] from Bx_ptr mapped as (b, ci, t)
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_T_ptr,        # *f32, shape (B, 3H, S)
    conv_out_ptr,     # *f32, shape (B, H, S)
    y_ptr,            # *f32, shape (B, S, H)
    B, S, H, Nproj,   # ints
    stride_bcx_b, stride_bcx_ch, stride_bcx_s,
    stride_conv_b, stride_conv_ci, stride_conv_t,
    stride_y_b, stride_y_s, stride_y_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Read C from channel 2
    C_val = tl.load(BCx_T_ptr + b * stride_bcx_b + 2 * stride_bcx_ch + s * stride_bcx_s)
    conv_val = tl.load(conv_out_ptr + b * stride_conv_b + h * stride_conv_ci + s * stride_conv_t)

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
    out_ptr,               # *f32, shape (B, S, H)
    B, S, H,               # ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co_out = tl.program_id(2)  # output channel index (also hidden_size index)

    if b >= B or s >= S or co_out >= H:
        return

    acc = 0.0
    for i in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + i * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + co_out * stride_w_o + i * stride_w_i)
        acc += y_val * w_val
    b_val = tl.load(out_proj_bias_ptr + co_out)
    acc += b_val

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + co_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguity
        device = x.device
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        Nproj = 3 * H

        # 1) Triple linear projection: BCx (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=device)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Transpose BCx to (B, 3H, S) for easy channel access
        BCx_T = BCx.transpose(1, 2).contiguous()  # (B, 3H, S)

        # 3) Bx = B * x_proj via gating kernel
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_T, Bx,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Grouped causal conv1d on Bx with kernel_size=4, groups=H
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Gating with C: y = C * conv_out
        y = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx_T, conv_out, y,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 6) Final linear projection
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
