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
    B, S, H, Nproj,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
    num_warps: tl.constexpr,
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


# Kernel 2: Gating Bx = B * x_proj
# BCx_T: (B, 3H, S) where channels 0,1,2 correspond to B, x_proj, C respectively.
# Output Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_T_ptr,  # *f32, shape (B, 3H, S)
    Bx_ptr,     # *f32, shape (B, S, H)
    B, S, H, Nproj,
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_bx_b, stride_bx_s, stride_bx_h,
    num_warps: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index corresponds to h in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # Read B and x_proj from channels 0 and 1
    B_val = tl.load(BCx_T_ptr + b * stride_bc_b + 0 * stride_bc_c + s * stride_bc_s)
    x_proj_val = tl.load(BCx_T_ptr + b * stride_bc_b + 1 * stride_bc_c + s * stride_bc_s)

    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,              # *f32, shape (B, S, H)
    out_proj_weight_ptr,# *f32, shape (H, H)
    out_proj_bias_ptr,  # *f32, shape (H,)
    out_ptr,            # *f32, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_oh, stride_w_oi,
    stride_out_b, stride_out_s, stride_out_h,
    num_warps: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    oh = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or oh >= H:
        return

    acc = 0.0
    for oi in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + oi * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + oh * stride_w_oh + oi * stride_w_oi)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + oh)
    acc += bias_val

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + oh * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous for Triton kernels
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        Nproj = 3 * H

        x32 = x.contiguous().to(dtype)
        in_proj_weight32 = in_proj_weight.contiguous().to(dtype)
        in_proj_bias32 = in_proj_bias.contiguous().to(dtype)
        conv_weight32 = conv_weight.contiguous().to(dtype)
        conv_bias32 = conv_bias.contiguous().to(dtype)
        out_proj_weight32 = out_proj_weight.contiguous().to(dtype)
        out_proj_bias32 = out_proj_bias.contiguous().to(dtype)

        # 1) Triple linear projection: BCx (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), dtype=dtype, device=device)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x32, in_proj_weight32, in_proj_bias32, BCx,
            B, S, H, Nproj,
            x32.stride(0), x32.stride(1), x32.stride(2),
            in_proj_weight32.stride(0), in_proj_weight32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1,
        )

        # Transpose to (B, 3H, S) for easy channel access
        BCx_T = BCx.transpose(1, 2).contiguous()  # (B, 3H, S)

        # 2) Gating: Bx = B * x_proj -> Bx: (B, S, H)
        Bx = torch.empty((B, S, H), dtype=dtype, device=device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_T, Bx,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1,
        )

        # 3) Grouped causal 1D convolution via PyTorch to ensure correctness
        # padding=kernel_size-1 for causal, groups=H
        # conv expects input (B, C_in, L_in) = (B, H, S), weight (C_out, C_in, K) = (H, H, 4)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad left by 3
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight32, conv_bias32, stride=1, padding=0, groups=H
        )  # (B, H, S)

        # 4) Gating with C: y = C * conv_out
        # Extract C from BCx_T channel 2: C = BCx_T[:, 2, :]
        C_vec = BCx_T[:, 2, :]  # (B, S)
        y = torch.empty((B, S, H), dtype=dtype, device=device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx_T, conv_out, y,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1,
        )

        # 5) Final output projection
        output = torch.empty((B, S, H), dtype=dtype, device=device)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y, out_proj_weight32, out_proj_bias32, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight32.stride(0), out_proj_weight32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
