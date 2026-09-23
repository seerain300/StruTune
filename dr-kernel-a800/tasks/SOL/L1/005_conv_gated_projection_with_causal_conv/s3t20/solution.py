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
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val
    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating: Bx = B * x_proj
# We reconstruct B (co=0) and x_proj (co=1) from BCx (B, S, 3H) and write (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                # *f32, shape (B, S, 3H)
    Bx_ptr,                 # *f32, shape (B, S, H)
    B, S, H,                # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# We implement padding explicitly for causal: for t=0, only k=0 contributes; for t=1, k=0,1; etc.
# Bx: (B, S, H), conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_s, stride_bx_h,  # strides for (b, s, h) on Bx
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (groups=H)
    if b >= B or ci >= H:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # For each time position t, compute y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            # Masked load: if x_pos is in range, load; else treat as 0 (causal padding)
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store to conv_out at position (b, ci, t)
    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_t)
    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_go, stride_w_gi,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for gi in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + gi * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_go + gi * stride_w_gi)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val
    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and float32, contiguous
        device = x.device
        B, S, H = x.shape
        assert conv_weight.shape[0] == H and conv_weight.shape[2] == 4, "conv_weight must be (H, H, 4)"
        Nproj = 3 * H

        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        # 1) Triple linear projection: BCx (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), device=device, dtype=torch.float32)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx, B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv on Bx with kernel_size=4 (groups=H). Explicit padding for causal.
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
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
        y = torch.empty((B, S, H), device=device, dtype=torch.float32)
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
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
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
