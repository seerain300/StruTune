import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), float32
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


# Kernel 2: Gating: Bx = B * x_proj
# BCx is the triple projection (B, S, 3H), we read channels 0 and 1 to form B and x_proj.
# Output Bx: (B, S, H), float32
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
    h = tl.program_id(2)  # channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # B is channel 0, x_proj is channel 1 in BCx
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    B, S, H,                # ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_or, stride_w_oi,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    if b >= B or s >= S or h_out >= H:
        return

    acc = 0.0
    for oi in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + oi * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_or + oi * stride_w_oi)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val
    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized forward. All heavy computation except grouped conv1d is performed by Triton kernels.
        Convolution is done via PyTorch F.conv1d for robustness and correctness.
        """
        # Ensure dtype is float32 for Triton kernels and shapes
        B, S, H = x.shape
        Nproj = 3 * H
        device = x.device

        # 1) Triple linear projection: BCx (B, S, 3H)
        x_f = x.contiguous().to(torch.float32)                      # (B, S, H)
        in_proj_w_f = in_proj_weight.contiguous().to(torch.float32) # (Nproj, H)
        in_proj_b_f = in_proj_bias.contiguous().to(torch.float32)   # (Nproj,)

        BCx = torch.empty((B, S, Nproj), device=device, dtype=torch.float32)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x_f, in_proj_w_f, in_proj_b_f, BCx,
            B, S, H, Nproj,
            x_f.stride(0), x_f.stride(1), x_f.stride(2),
            in_proj_w_f.stride(0), in_proj_w_f.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: reconstruct B and x_proj from BCx and compute Bx
        # BCx has channels: B at 0, x_proj at 1, and unused at 2..3H-1
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D convolution via PyTorch conv1d (robust, correct)
        # Bx: (B, H, S) conceptual; conv_weight: (H, H, 4)
        # Padding for causal: pad left by 3 (kernel_size-1)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S + 3)

        conv_weight_f = conv_weight.contiguous().to(torch.float32)  # (H, H, 4)
        conv_bias_f = conv_bias.contiguous().to(torch.float32)      # (H,)

        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight_f, conv_bias_f, groups=H, stride=1, padding=0
        )  # (B, H, S)

        # 4) Gating with C: y = C * conv_out
        # Read C from BCx[:, 2, :] which corresponds to channel index 2
        C_vec = BCx[:, :, 2].contiguous()  # shape (B, S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)

        # We need to multiply elementwise: y[b, ci, t] = C[b, t] * conv_out[b, ci, t]
        # Broadcast C_vec over ci dimension
        for ci in range(0, H):
            conv_ci = conv_out[:, ci, :]  # (B, S)
            y[:, ci, :] = conv_ci * C_vec

        # 5) Final linear projection
        out_proj_w_f = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_b_f = out_proj_bias.contiguous().to(torch.float32)    # (H,)

        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid4 = (B, S, H)
        linear_final_kernel[grid4](
            y, out_proj_w_f, out_proj_b_f, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w_f.stride(0), out_proj_w_f.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
