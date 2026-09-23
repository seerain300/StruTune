import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H) float32
# in_proj_weight: (Nproj, H) float32, Nproj = 3 * H
# in_proj_bias: (Nproj,) float32
# output: BCx (B, S, Nproj) float32
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, shape (B, S, H)
    in_proj_weight_ptr,     # *f32, shape (Nproj, H)
    in_proj_bias_ptr,       # *f32, shape (Nproj,)
    BCx_ptr,                # *f32, shape (B, S, Nproj)
    B, S, H, Nproj,         # runtime ints
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
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val

    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating mul on transposed BCx
# We conceptually use BCx as (B, 3H, S): B at channel 0, x_proj at channel 1.
# Output: Bx (B, H, S) float32 where Bx[b, h, s] = B[b, 0, s] * x_proj[b, 1, s]
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                # *f32, shape (B, S, 3H) but we index as (b, c, t) via mapping
    Bx_ptr,                 # *f32, shape (B, H, S)
    B, S, H,                # runtime ints
    stride_bc_b, stride_bc_s, stride_bc_co,  # strides for BCx
    stride_bx_b, stride_bx_ci, stride_bx_t,  # strides for Bx (b, ci, t)
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # time position
    ci = tl.program_id(2) # channel index in [0, H)

    if b >= B or t >= S or ci >= H:
        return

    # Read B at channel 0 and x_proj at channel 1 of BCx, which is (B, S, 3H)
    B_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 1 * stride_bc_co)

    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + t * stride_bx_t, bx)


# Kernel 3: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H) float32
# out_proj_weight: (H, H) float32
# out_proj_bias: (H,) float32
# output: (B, S, H) float32
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    B, S, H,                # runtime ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_r, stride_w_c,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for c in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_r + c * stride_w_c)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton forward using Triton for triple linear, gating, and final linear.
        Conv is done via PyTorch for correctness (grouped causal 1D, kernel_size=4).
        """
        # Cast all inputs/params to float32 for Triton kernels
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        batch_size, seq_len, hidden_size = x.shape
        Nproj = in_proj_weight.shape[0]
        H = hidden_size

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((batch_size, seq_len, Nproj), device=x.device, dtype=torch.float32)

        grid1 = (batch_size, seq_len, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            batch_size, seq_len, hidden_size, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Elementwise gating: Bx = B * x_proj
        # Read B at channel 0 and x_proj at channel 1 of BCx (conceptually as (B, 3H, S))
        Bx = torch.empty((batch_size, hidden_size, seq_len), device=x.device, dtype=torch.float32)

        grid2 = (batch_size, seq_len, hidden_size)
        gating_mul_kernel[grid2](
            BCx, Bx,
            batch_size, seq_len, hidden_size,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D conv on Bx: conv1d in PyTorch for correctness (groups=H, kernel_size=4)
        # Pad left by 3 for causal (kernel_size=4 => pad=kernel-1)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, groups=hidden_size, stride=1, padding=0, dilation=1, bias=conv_bias
        )  # shape: (B, H, S)

        # 4) Output gating: y = C * conv_out, where C corresponds to channels 2H..3H-1 in BCx
        # Construct C tensor from BCx: C[b, c, t] = BCx[b, t, 2*H + c]
        C_for_gating = torch.empty((batch_size, hidden_size, seq_len), device=x.device, dtype=torch.float32)

        @triton.jit
        def extract_C_kernel(
            BCx_ptr,           # *f32, shape (B, S, 3H)
            C_ptr,             # *f32, shape (B, H, S)
            B, S, H,           # runtime ints
            stride_bc_b, stride_bc_s, stride_bc_co,  # strides for BCx
            stride_c_b, stride_c_ci, stride_c_t,     # strides for C
        ):
            b = tl.program_id(0)
            t = tl.program_id(1)
            c = tl.program_id(2)
            if b >= B or t >= S or c >= H:
                return
            channel = 2 * H + c
            val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + channel * stride_bc_co)
            tl.store(C_ptr + b * stride_c_b + c * stride_c_ci + t * stride_c_t, val)

        grid_extract_C = (batch_size, seq_len, hidden_size)
        extract_C_kernel[grid_extract_C](
            BCx, C_for_gating,
            batch_size, seq_len, hidden_size,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            C_for_gating.stride(0), C_for_gating.stride(1), C_for_gating.stride(2),
            num_warps=1, num_stages=1,
        )

        # Multiply y = C_for_gating * conv_out
        y = C_for_gating * conv_out  # shape (B, H, S)

        # 5) Final linear projection
        output = torch.empty((batch_size, seq_len, hidden_size), device=x.device, dtype=torch.float32)

        grid3 = (batch_size, seq_len, hidden_size)
        linear_final_kernel[grid3](
            y, out_proj_weight, out_proj_bias, output,
            batch_size, seq_len, hidden_size,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
