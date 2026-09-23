import torch
import triton
import triton.language as tl


def _ptr_type_from_tensor(t: torch.Tensor):
    # Return Triton pointer type corresponding to tensor dtype
    if t.dtype == torch.float32:
        return tl.pointer_type(tl.float32)
    elif t.dtype == torch.float16:
        return tl.pointer_type(tl.float16)
    elif t.dtype == torch.bfloat16:
        return tl.pointer_type(tl.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype: {t.dtype}")


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), same dtype as x
@triton.jit
def triple_linear_kernel(
    x_ptr: _ptr_type,                   # *T, shape (B, S, H)
    in_proj_weight_ptr: _ptr_type,      # *T, shape (Nproj, H)
    in_proj_bias_ptr: _ptr_type,        # *T, shape (Nproj,)
    BCx_ptr: _ptr_type,                 # *T, shape (B, S, Nproj)
    B, S,                               # runtime ints
    H: tl.constexpr,                    # compile-time constant for loop
    Nproj: tl.constexpr,                # compile-time constant for output channels
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


# Kernel 2: Gating: Bx = B * x_proj
# Input BCx is (B, S, 3H), conceptual split into B (channel 0) and x_proj (channel 1).
@triton.jit
def gating_mul_kernel(
    BCx_ptr: _ptr_type,                # *T, shape (B, S, 3H)
    Bx_ptr: _ptr_type,                 # *T, shape (B, S, H)
    B, S,                              # runtime ints
    H: tl.constexpr,                   # compile-time constant for loop
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Read B from channel 0
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    # Read x_proj from channel 1
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: (B, H, S) conceptual indexing as (b, ci, t). We pass Bx as (B, S, H) and remap indices.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S), same dtype as Bx
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr: _ptr_type,             # *T, shape (B, S, H)
    conv_weight_ptr: _ptr_type,    # *T, shape (H, H, 4)
    conv_bias_ptr: _ptr_type,      # *T, shape (H,)
    conv_out_ptr: _ptr_type,       # *T, shape (B, H, S)
    B, S, H,                       # runtime ints
    stride_bx_b, stride_bx_s, stride_bx_h,  # for (b, s, h) on Bx
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output/input channel index
    t = tl.program_id(2)   # time index

    if b >= B or ci >= H or t >= S:
        return

    acc = 0.0
    for k in range(0, 4):
        x_pos = t + k
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr: _ptr_type,        # *T, shape (B, S, 3H), conceptual channel index is 2
    conv_out_ptr: _ptr_type,   # *T, shape (B, H, S)
    y_ptr: _ptr_type,          # *T, shape (B, S, H)
    B, S, H,                   # runtime ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index

    if b >= B or s >= S or h >= H:
        return

    # Read C from channel 2
    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    # Read conv_out[b, h, s] from conv_out (shape (B, H, S))
    co_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_t)
    y_val = C_val * co_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection y -> out_proj(y)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H), same dtype as y
@triton.jit
def linear_final_kernel(
    y_ptr: _ptr_type,                 # *T, shape (B, S, H)
    out_proj_weight_ptr: _ptr_type,   # *T, shape (H, H)
    out_proj_bias_ptr: _ptr_type,     # *T, shape (H,)
    out_ptr: _ptr_type,               # *T, shape (B, S, H)
    B, S,                             # runtime ints
    H: tl.constexpr,                  # compile-time constant for loop
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)  # output channel index

    if b >= B or s >= S or h_out >= H:
        return

    acc = 0.0
    for i in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + i * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_o + i * stride_w_i)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + h_out)
    acc += bias
    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias,
                conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Triton kernels use pointer types determined by x.dtype; ensure contiguity and same device.
        B, S, H = x.shape
        Nproj = 3 * H

        # 1) Triple linear: BCx (B, S, 3H), same dtype as x
        BCx = torch.empty((B, S, Nproj), dtype=x.dtype, device=x.device)
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

        # 2) Gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S,
            H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D convolution (kernel_size=4, groups=H)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        grid3 = (B, H, S)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),   # mapping: (b, s, h) on Bx
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
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

        # 5) Final linear projection
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
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
