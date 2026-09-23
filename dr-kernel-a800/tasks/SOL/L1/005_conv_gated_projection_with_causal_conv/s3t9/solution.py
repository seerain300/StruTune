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
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    acc = 0.0
    for ci in range(0, H):
        xi = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        wi = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += xi * wi
    # add bias
    bi = tl.load(in_proj_bias_ptr + co)
    acc += bi

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Elementwise multiply Bx = B * x_proj, reading from BCx_T (B, 3H, S) channels 0 and 1
@triton.jit
def gating_mul_kernel(
    BCxT_ptr,               # *f32, shape (B, 3H, S)
    Bx_ptr,                 # *f32, shape (B, S, H)
    B, S, H, Nproj,         # ints
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_bc_b, stride_bc_co, stride_bc_s,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index within H, 0..H-1

    if b >= B or s >= S or h >= H:
        return

    B_val = tl.load(BCxT_ptr + b * stride_bc_b + 0 * stride_bc_co + s * stride_bc_s)
    x_proj_val = tl.load(BCxT_ptr + b * stride_bc_b + 1 * stride_bc_co + s * stride_bc_s)
    bx = B_val * x_proj_val

    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D conv (depthwise, kernel_size=4, groups=H)
# Bx: conceptual (b, ci, t) with tensor (B, S, H) strides for (b, t, ci)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_s, stride_bx_h,  # for (b, s, h)
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_h, stride_out_s,  # for (b, h, s)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S <= 0:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            # Safe load with padding: if x_pos >= S, treat as zero (causal pad)
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_h + t * stride_out_s, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx_T[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCxT_ptr,        # *f32, shape (B, 3H, S)
    conv_out_ptr,    # *f32, shape (B, H, S)
    y_ptr,           # *f32, shape (B, S, H)
    B, S, H, Nproj,  # ints
    stride_bc_b, stride_bc_co, stride_bc_s,
    stride_out_b, stride_out_h, stride_out_s,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    C_val = tl.load(BCxT_ptr + b * stride_bc_b + 2 * stride_bc_co + s * stride_bc_s)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s)
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
    stride_w_ji, stride_w_jk,  # out_proj_weight strides (ji, jk)
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ho = tl.program_id(2)  # output channel index within H, 0..H-1

    if b >= B or s >= S or ho >= H:
        return

    acc = 0.0
    for j in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + j * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + j * stride_w_ji + ho * stride_w_jk)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + ho)
    acc += bias_val

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + ho * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Cast and make contiguous, ensure float32 for Triton
        device = x.device
        dtype = torch.float32
        B, S, H = x.shape
        Nproj = 3 * H

        x = x.contiguous().to(dtype)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)
        conv_weight = conv_weight.contiguous().to(dtype)
        conv_bias = conv_bias.contiguous().to(dtype)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)

        # 1) Triple linear projection: BCx (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), dtype=dtype, device=device)
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
        Bx = torch.empty((B, S, H), dtype=dtype, device=device)  # (B, S, H)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_T, Bx,
            B, S, H, Nproj,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv on Bx, kernel_size=4, groups=H
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=dtype, device=device)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
        y = torch.empty((B, S, H), dtype=dtype, device=device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx_T, conv_out, y,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection
        output = torch.empty((B, S, H), dtype=dtype, device=device)
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
