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
    Nproj: tl.int32, H: tl.int32,
    stride_x_b: tl.int32, stride_x_s: tl.int32, stride_x_h: tl.int32,
    stride_w_co: tl.int32, stride_w_ci: tl.int32,
    stride_bc_b: tl.int32, stride_bc_s: tl.int32, stride_bc_co: tl.int32,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    acc = 0.0
    # Loop over hidden dimension H (runtime-bound)
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val

    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating: Bx = B * x_proj, where B=BCx[:,0,:], x_proj=BCx[:,1,:]
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, shape (B, 3, S) logically (we read channels 0 and 1)
    Bx_ptr,         # *f32, shape (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    stride_bc_b: tl.int32, stride_bc_c: tl.int32, stride_bc_s: tl.int32,
    stride_bx_b: tl.int32, stride_bx_s: tl.int32, stride_bx_h: tl.int32,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    B_vec = tl.load(BCx_ptr + b * stride_bc_b + 0 * stride_bc_c + s * stride_bc_s)
    x_proj_vec = tl.load(BCx_ptr + b * stride_bc_b + 1 * stride_bc_c + s * stride_bc_s)

    bx = B_vec * x_proj_vec
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: conceptual (b, ci, t) where Bx is (B, S, H). We index via strides.
# conv_weight: (H, H, 4), conv_bias: (H,), conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B: tl.int32, S: tl.int32, H: tl.int32,
    stride_bx_b: tl.int32, stride_bx_ci: tl.int32, stride_bx_t: tl.int32,  # (b, ci, t) mapping on Bx
    stride_w_go: tl.int32, stride_w_gi: tl.int32, stride_w_k: tl.int32,    # conv_weight strides
    stride_out_b: tl.int32, stride_out_ci: tl.int32, stride_out_t: tl.int32,  # conv_out strides
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output/input channel index
    if b >= B or ci >= H or S <= 0:
        return

    acc = 0.0

    # y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            # Ensure x_pos in bounds
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, 3, S) logically (we read channel 2 for C)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    stride_bc_b: tl.int32, stride_bc_c: tl.int32, stride_bc_s: tl.int32,
    stride_out_b: tl.int32, stride_out_ci: tl.int32, stride_out_t: tl.int32,
    stride_y_b: tl.int32, stride_y_s: tl.int32, stride_y_h: tl.int32,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    C_val = tl.load(BCx_ptr + b * stride_bc_b + 2 * stride_bc_c + s * stride_bc_s)
    out_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_ci + s * stride_out_t)
    y_val = C_val * out_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,), output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    H: tl.int32,
    stride_y_b: tl.int32, stride_y_s: tl.int32, stride_y_h: tl.int32,
    stride_w_out_co: tl.int32, stride_w_out_ci: tl.int32,
    stride_out_b: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = 0.0
    for h_in in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h_in * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_out_co + h_in * stride_w_out_ci)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure float32 and contiguity for Triton kernels
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]
        assert in_proj_weight.shape[1] == H, "in_proj_weight second dim must equal hidden_size"
        assert in_proj_bias.shape[0] == Nproj, "in_proj_bias shape mismatch"
        assert conv_weight.shape[0] == H and conv_weight.shape[1] == H and conv_weight.shape[2] == 4, "conv_weight must be (H, H, 4)"
        assert conv_bias.shape[0] == H, "conv_bias shape mismatch"
        assert out_proj_weight.shape[0] == H and out_proj_weight.shape[1] == H, "out_proj_weight must be (H, H)"
        assert out_proj_bias.shape[0] == H, "out_proj_bias shape mismatch"

        # Cast to float32 and contiguous
        x = x.contiguous().to(dtype)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)
        conv_weight = conv_weight.contiguous().to(dtype)
        conv_bias = conv_bias.contiguous().to(dtype)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)

        # 1) Triple linear projection: BCx shape (B, S, Nproj), logically (B, 3, S) with channels [B, C, x_proj]
        BCx = torch.empty((B, S, Nproj), device=device, dtype=dtype)

        grid_tl = (B, S, Nproj)
        triple_linear_kernel[grid_tl](
            x, in_proj_weight, in_proj_bias, BCx,
            Nproj, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=dtype)

        grid_g = (B, S, H)
        gating_mul_kernel[grid_g](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv1d on Bx: conv_out shape (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_conv = (B, H)
        causal_conv_groups_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # treat Bx as (b, ci, t)
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Output gating: y = C * conv_out; C comes from BCx[:, 2, :]
        y = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_y = (B, S, H)
        gating_mul_y_kernel[grid_y](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection
        output = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_final = (B, S, H)
        linear_final_kernel[grid_final](
            y, out_proj_weight, out_proj_bias, output,
            H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
