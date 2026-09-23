import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), dtype = x.dtype
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *T, shape (B, S, H)
    in_proj_weight_ptr,     # *T, shape (Nproj, H)
    in_proj_bias_ptr,       # *T, shape (Nproj,)
    BCx_ptr,                # *T, shape (B, S, Nproj)
    B, S, H, Nproj,         # ints
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    # Map co to (c, p): c = co // 3, p = co % 3
    c = co // 3
    p = co % 3

    acc = 0.0
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + c * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val

    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating: Bx = B * x_proj
# BCx: (B, 3H, S), read B from channel 0, x_proj from channel 1, write Bx (B, H, S)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                # *T, shape (B, 3H, S)
    Bx_ptr,                 # *T, shape (B, H, S)
    B, S, H,                # ints
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_bx_b, stride_bx_ci, stride_bx_t,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    B_val = tl.load(BCx_ptr + b * stride_bc_b + 0 * stride_bc_c + s * stride_bc_s)
    x_val = tl.load(BCx_ptr + b * stride_bc_b + 1 * stride_bc_c + s * stride_bc_s)

    bx = B_val * x_val
    tl.store(Bx_ptr + b * stride_bx_b + h * stride_bx_ci + s * stride_bx_t, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: (B, S, H) conceptual indexing as (b, ci, t) is mapped via strides.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *T, shape (B, S, H)
    conv_weight_ptr,    # *T, shape (H, H, 4)
    conv_bias_ptr,      # *T, shape (H,)
    conv_out_ptr,       # *T, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_s, stride_bx_h,  # strides for (b, s, h) on Bx
    stride_w_go, stride_w_gi, stride_w_k,    # strides for (go, gi, k) on conv_weight
    stride_out_b, stride_out_h, stride_out_s,  # strides for (b, ci, t) on conv_out
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S <= 0:
        return

    acc = 0.0
    # Initialize accumulator for this (b, ci)
    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for k in range(0, 4):
        dot = 0.0
        for t in range(0, S):
            x_pos = t + k
            # Masked load for causal padding
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            dot += x_val * w_val
        acc += dot

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_h + 0 * stride_out_s, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *T, shape (B, 3H, S)
    conv_out_ptr,   # *T, shape (B, H, S)
    y_ptr,          # *T, shape (B, S, H)
    B, S, H,
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index

    if b >= B or s >= S or h >= H:
        return

    C_val = tl.load(BCx_ptr + b * stride_bc_b + 2 * stride_bc_c + s * stride_bc_s)
    conv_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_t)

    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# out: (B, S, H), dtype = y.dtype
@triton.jit
def final_linear_kernel(
    y_ptr,                 # *T, shape (B, S, H)
    out_proj_weight_ptr,   # *T, shape (H, H)
    out_proj_bias_ptr,     # *T, shape (H,)
    out_ptr,               # *T, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_ci, stride_w_co,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    if b >= B or s >= S or h_out >= H:
        return

    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + ci * stride_w_ci + h_out * stride_w_co)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure contiguity without dtype casting; preserve input dtype
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]
        assert Nproj == 3 * H, "in_proj_weight must have shape (3*H, H)"

        # Step 1: Triple linear projection -> BCx shape (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), dtype=x.dtype, device=x.device)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 2: Gating Bx = B * x_proj
        # Read B and x_proj from BCx[:, 0, :], BCx[:, 1, :]
        Bx = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx -> conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # (b, s, h) on Bx
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 4: Gating y = C * conv_out, read C from BCx[:, 2, :]
        y = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 5: Final linear projection -> output (B, S, H)
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid5 = (B, S, H)
        final_linear_kernel[grid5](
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
