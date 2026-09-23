import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H) float32
# in_proj_weight: (Nproj, H) float32, Nproj = 3 * H
# in_proj_bias: (Nproj,) float32
# BCx_out: (B, S, Nproj) float32
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
    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating Bx = B * x_proj, using BCx where:
#   B = BCx[:, :, 0], x_proj = BCx[:, :, 1]
# Output Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    Bx_ptr,         # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    # Read B and x_proj from BCx channels 0 and 1
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    bx_val = B_val * x_proj_val

    # Store to Bx at channel h
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx_val)


# Kernel 3: Grouped causal 1D conv (depthwise, kernel_size=4) on Bx
# Bx: conceptual (B, H, S_in) via strides; we launch grid over (B, H) and iterate over output t and k.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S_out), where S_out = S_in - 3 (causal pad of 3 zeros)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S_in, H) conceptual indexing (b, ci, t)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S_out)
    B, S_in, H,         # ints
    S_out,              # output length = S_in - 3
    stride_bx_b, stride_bx_ci, stride_bx_t,   # strides for Bx
    stride_w_go, stride_w_gi, stride_w_k,     # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t, # strides for conv_out
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S_out <= 0:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S_out):
        for k in range(0, 4):
            x_pos = t + k
            if x_pos < S_in:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
                w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store conv_out[b, ci, t]
    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, :, 2]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S_out)
    y_ptr,          # *f32, shape (B, S_out, H)
    B, S, H, S_out,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or t >= S_out or h >= H:
        return

    # Read C from BCx[:, :, 2]
    C_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 2 * stride_bc_co)
    # Read conv_out[b, h, t]
    co_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + t * stride_co_t)
    y_val = C_val * co_val

    # Store to y[b, t, h]
    tl.store(y_ptr + b * stride_y_b + t * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S_out, H) float32
# out_proj_weight: (H, H) float32
# out_proj_bias: (H,) float32
# output: (B, S_out, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S_out, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S_out, H)
    B, S_out, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_row, stride_w_col,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S_out or h >= H:
        return

    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_row + ci * stride_w_col)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        All computation must be performed via Triton kernels launched here.
        We avoid any torch ops (no F.linear, no conv1d, no elementwise torch ops).
        """
        # Ensure float32 and contiguous
        x32 = x.contiguous().to(torch.float32)
        in_proj_w32 = in_proj_weight.contiguous().to(torch.float32)
        in_proj_b32 = in_proj_bias.contiguous().to(torch.float32)

        # 1) Triple linear projection: BCx (B, S, 3H)
        B, S, H = x32.shape
        Nproj = 3 * H
        BCx = torch.empty((B, S, Nproj), device=x32.device, dtype=torch.float32)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x32, in_proj_w32, in_proj_b32, BCx,
            B, S, H, Nproj,
            x32.stride(0), x32.stride(1), x32.stride(2),
            in_proj_w32.stride(0), in_proj_w32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        # We reconstruct B and x_proj from BCx by reading channels 0 and 1
        Bx = torch.empty((B, S, H), device=x32.device, dtype=torch.float32)

        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv on Bx with kernel_size=4 and groups=H.
        # Implement left causal padding by working on output length S_out = S - 3
        S_out = S - 3
        conv_out = torch.empty((B, H, S_out), device=x32.device, dtype=torch.float32)

        grid3 = (B, H)
        conv_w32 = conv_weight.contiguous().to(torch.float32)
        conv_b32 = conv_bias.contiguous().to(torch.float32)

        causal_conv_groups_kernel[grid3](
            Bx, conv_w32, conv_b32, conv_out,
            B, S, H, S_out,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),   # (b, ci, t) strides for Bx
            conv_w32.stride(0), conv_w32.stride(1), conv_w32.stride(2),  # (go, gi, k)
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),  # (b, ci, t)
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
        y = torch.empty((B, S_out, H), device=x32.device, dtype=torch.float32)

        grid4 = (B, S_out, H)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H, S_out,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final output projection: F.linear(y, out_proj_weight, out_proj_bias)
        out_proj_w32 = out_proj_weight.contiguous().to(torch.float32)
        out_proj_b32 = out_proj_bias.contiguous().to(torch.float32)

        output = torch.empty((B, S_out, H), device=x32.device, dtype=torch.float32)

        grid5 = (B, S_out, H)
        linear_final_kernel[grid5](
            y, out_proj_w32, out_proj_b32, output,
            B, S_out, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w32.stride(0), out_proj_w32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
