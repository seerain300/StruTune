import torch
import triton
import triton.language as tl


def cast_tensors(*tensors):
    # Cast all provided tensors to float32 (for numerical robustness and Triton compatibility)
    result = []
    for t in tensors:
        if t is not None:
            result.append(t.to(torch.float32).contiguous())
        else:
            result.append(None)
    return tuple(result)


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


# Kernel 2: Gating Bx = B * x_proj where B = BCx[:, 0, :], x_proj = BCx[:, 1, :], result shape (B, S, 1)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                # *f32, shape (B, S, 3H)
    Bx_ptr,                 # *f32, shape (B, S, 1)
    B, S, Nproj,            # ints (runtime)
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_co,  # co=0 always, we keep co to match (B, S, 1)
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # should be 0

    if b >= B or s >= S or co >= 1:
        return

    # Read B and x_proj as scalars
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx_val = B_val * x_proj_val

    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + co * stride_bx_co, bx_val)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx is conceptual (B, 1, S): Bx_ptr is actually (B, S, 1) but we index as (b, ci=0, t=s)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, 1)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_co, stride_bx_t,  # for (b, co, t) where co=0, t=s
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    t = tl.program_id(2)   # time position

    if b >= B or ci >= H or t >= S:
        return

    acc = 0.0
    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, 0, t + k] + bias[ci]
    for k in range(0, 4):
        x_pos = t + k
        if x_pos < S:
            # Bx_ptr maps as (b, co=0, t=x_pos) which is index (b, x_pos, 0)
            x_val = tl.load(Bx_ptr + b * stride_bx_b + 0 * stride_bx_co + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; C is (B, S, 1) from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, H, S)
    B, S, Nproj, H, # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_out_b, stride_out_ci, stride_out_t,
    stride_y_b, stride_y_ci, stride_y_t,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index
    t = tl.program_id(2)   # time position

    if b >= B or ci >= H or t >= S:
        return

    C_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 2 * stride_bc_co)  # C is at co=2
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t)
    y_val = C_val * conv_val

    tl.store(y_ptr + b * stride_y_b + ci * stride_y_ci + t * stride_y_t, y_val)


# Kernel 5: Final linear projection y -> output
# y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H,), output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                 # *f32, shape (B, H, S)
    out_proj_weight_ptr,   # *f32, shape (H, H)
    out_proj_bias_ptr,     # *f32, shape (H,)
    output_ptr,            # *f32, shape (B, S, H)
    B, S, H,               # ints
    stride_y_b, stride_y_ci, stride_y_t,
    stride_w_row, stride_w_col,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index (same as input feature)

    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    # y[b, h, s] is a scalar, out_proj_weight[h, :] is a vector of length H
    for i in range(0, H):
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_row + i * stride_w_col)
        y_val = tl.load(y_ptr + b * stride_y_b + i * stride_y_ci + s * stride_y_t)  # y[b, i, s]
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Cast all tensors to float32 to ensure Triton compatibility and numerical robustness
        x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias = cast_tensors(
            x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias
        )

        B, S, H = x.shape
        Nproj = 3 * H

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

        # Step 2: Gating Bx = B * x_proj, result shape (B, S, 1)
        Bx = torch.empty((B, S, 1), dtype=x.dtype, device=x.device)

        grid2 = (B, S, 1)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, Nproj,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 3: Grouped causal conv1d on Bx, kernel_size=4, groups=H -> conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)

        grid3 = (B, H, S)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(2), 1,  # co=0, t=s; stride_bc_co unused (we hard-coded co=0)
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 4: Gating y = C * conv_out, C = BCx[:, 2, :] shape (B, S, 1)
        y = torch.empty((B, H, S), dtype=x.dtype, device=x.device)

        grid4 = (B, H, S)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, Nproj, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # Step 5: Final linear projection -> output (B, S, H)
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)

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
