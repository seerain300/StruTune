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
    b = tl.load(in_proj_bias_ptr + co)
    acc += b
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating Bx = B * x_proj. Read from BCx_T (B, 3H, S), channels 0 and 1.
# Output Bx (B, S, H).
@triton.jit
def gating_mul_kernel(
    BCx_T_ptr,              # *f32, shape (B, 3H, S)
    Bx_ptr,                 # *f32, shape (B, S, H)
    B, S, H, Nproj,         # ints
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_bcx_b, stride_bcx_c, stride_bcx_s,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    B_val = tl.load(BCx_T_ptr + b * stride_bcx_b + 0 * stride_bcx_c + s * stride_bcx_s)
    xproj_val = tl.load(BCx_T_ptr + b * stride_bcx_b + 1 * stride_bcx_c + s * stride_bcx_s)
    bx = B_val * xproj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Elementwise gating y = C * conv_out. Read C from BCx_T[:, 2, :], conv_out is (B, H, S).
@triton.jit
def gating_mul_y_kernel(
    BCx_T_ptr,              # *f32, shape (B, 3H, S)
    conv_out_ptr,           # *f32, shape (B, H, S)
    y_ptr,                  # *f32, shape (B, S, H)
    B, S, H, Nproj,         # ints
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    C_val = tl.load(BCx_T_ptr + b * stride_bc_b + 2 * stride_bc_c + s * stride_bc_s)
    conv_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_t)
    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 4: Final linear projection: output[b, s, co] = sum_h y[b, s, h] * out_proj_w[co, h] + out_proj_bias[co]
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    B, S, H,                # ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_co, stride_w_ci,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or co >= H:
        return

    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += y_val * w_val
    b = tl.load(out_proj_bias_ptr + co)
    acc += b
    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + co * stride_out_h, acc)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Cast to float32 for Triton kernels and ensure contiguity
        device = x.device
        B, S, H = x.shape
        Nproj = 3 * H

        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

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

        # 3) Compute Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_T, Bx,
            B, S, H, Nproj,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Grouped causal conv1d using PyTorch for robustness (matches original semantics)
        # Padding for causal conv: pad = kernel_size - 1
        pad = conv_weight.shape[2] - 1
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # pad only on the left
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, stride=1, padding=pad, groups=H
        )  # (B, H, S)

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
