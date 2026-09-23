import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), stored as float32
@triton.jit
def triple_linear_kernel(
    x_ptr: tl.pointer[tl.float32],              # *f32, shape (B, S, H)
    in_proj_weight_ptr: tl.pointer[tl.float32], # *f32, shape (Nproj, H)
    in_proj_bias_ptr: tl.pointer[tl.float32],   # *f32, shape (Nproj,)
    BCx_ptr: tl.pointer[tl.float32],            # *f32, shape (B, S, Nproj)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    # Accumulator for dot product
    acc = 0.0

    # Reduction over H dimension
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val

    # Add bias
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    # Store result
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating Bx = B * x_proj
# BCx: (B, S, 3H). We read channels 0 and 1: B is BCx[..., 0], x_proj is BCx[..., 1].
# Output Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr: tl.pointer[tl.float32],    # *f32, shape (B, S, 3H)
    Bx_ptr: tl.pointer[tl.float32],     # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b, stride_bc_s, stride_bc_c,  # strides for BCx
    stride_bx_b, stride_bx_s, stride_bx_h,  # strides for Bx
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Reconstruct B and x_proj from BCx
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_c)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_c)

    bx = B_val * x_proj_val

    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx conceptual indexing as (b, ci, t): we use Bx_ptr of shape (B, S, H) and index as (b, ci, t) via strides.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S), conceptual indexing (b, ci, t)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr: tl.pointer[tl.float32],       # *f32, shape (B, S, H)
    conv_weight_ptr: tl.pointer[tl.float32],  # *f32, shape (H, H, 4)
    conv_bias_ptr: tl.pointer[tl.float32],    # *f32, shape (H,)
    conv_out_ptr: tl.pointer[tl.float32],     # *f32, shape (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bx_b, stride_bx_ci, stride_bx_t,  # mapping for (b, ci, t) on Bx
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
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
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            else:
                # pad with 0 for out-of-range (conceptually padded zeros for causal)
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store to (b, ci, t)
    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr: tl.pointer[tl.float32],    # *f32, shape (B, S, 3H)
    conv_out_ptr: tl.pointer[tl.float32],  # *f32, shape (B, H, S) conceptual (b, ci, t)
    y_ptr: tl.pointer[tl.float32],      # *f32, shape (B, S, H) conceptual (b, t, h) but we store as (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b, stride_bc_s, stride_bc_c,    # BCx strides
    stride_out_b, stride_out_ci, stride_out_t,  # conv_out strides
    stride_y_b, stride_y_s, stride_y_h,       # y strides
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Reconstruct C from BCx[:, 2, :]
    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_c)

    # Load conv_out for channel h at time s
    out_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_ci + s * stride_out_t)

    y_val = C_val * out_val

    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr: tl.pointer[tl.float32],              # *f32, shape (B, S, H)
    out_proj_weight_ptr: tl.pointer[tl.float32],# *f32, shape (H, H)
    out_proj_bias_ptr: tl.pointer[tl.float32],  # *f32, shape (H,)
    output_ptr: tl.pointer[tl.float32],         # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_co, stride_w_ci,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output feature index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_co + ci * stride_w_ci)
        acc += y_val * w_val

    bias = tl.load(out_proj_bias_ptr + h)
    acc += bias

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:

        # Cast all inputs/params to float32 to avoid Triton dtype issues
        device = x.device
        B, S, H = x.shape
        Nproj = 3 * H

        x_f32 = x.contiguous().to(torch.float32)
        in_proj_weight_f32 = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias_f32 = in_proj_bias.contiguous().to(torch.float32)
        conv_weight_f32 = conv_weight.contiguous().to(torch.float32)
        conv_bias_f32 = conv_bias.contiguous().to(torch.float32)
        out_proj_weight_f32 = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias_f32 = out_proj_bias.contiguous().to(torch.float32)

        # Allocate BCx: (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=device)

        # Launch triple linear kernel: grid over (B, S, Nproj)
        grid_tl = (B, S, Nproj)
        triple_linear_kernel[grid_tl](
            x_f32, in_proj_weight_f32, in_proj_bias_f32, BCx,
            B, S, H, Nproj,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            in_proj_weight_f32.stride(0), in_proj_weight_f32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Reconstruct B and x_proj from BCx and compute Bx = B * x_proj
        # BCx is (B, S, 3H). We want Bx of shape (B, S, H) where h in [0, H)
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm = (B, S, H)
        gating_mul_kernel[grid_gm](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Grouped causal conv1d on Bx with kernel_size=4, groups=H
        # We conceptually index Bx as (b, ci, t), conv_out as (b, ci, t)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid_conv = (B, H)
        causal_conv_groups_kernel[grid_conv](
            Bx, conv_weight_f32, conv_bias_f32, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # strides for (b, ci, t) on Bx
            conv_weight_f32.stride(0), conv_weight_f32.stride(1), conv_weight_f32.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Gating with C: y = C * conv_out, reading C from BCx[:, 2, :]
        # Allocate y of shape (B, S, H)
        y = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm_y = (B, S, H)
        gating_mul_y_kernel[grid_gm_y](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # Final linear projection: y -> (B, S, H)
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_fin = (B, S, H)
        linear_final_kernel[grid_fin](
            y, out_proj_weight_f32, out_proj_bias_f32, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight_f32.stride(0), out_proj_weight_f32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
