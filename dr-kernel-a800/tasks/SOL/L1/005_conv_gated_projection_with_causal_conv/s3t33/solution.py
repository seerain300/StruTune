import torch
import triton
import triton.language as tl


# Kernel: Elementwise gating from BCx_t view, reconstruct B and x_proj from transposed view
# BCx_t: (B, 3H, S) as a non-contiguous view. We slice channels in host code and pass (B, S, H) tensors to kernel.
@triton.jit
def gating_mul_kernel_BCx_t(
    BCx_ptr,        # *f32, shape (B, S, H) derived from BCx_t slice
    Bx_ptr,         # *f32, shape (B, S, H) output
    B, S, H,        # ints
    stride_b, stride_s, stride_h,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    for h in range(0, H):
        B_val = tl.load(BCx_ptr + b * stride_b + s * stride_s + h * stride_h)
        x_val = tl.load(BCx_ptr + b * stride_b + s * stride_s + h * stride_h + H)  # next channel slice?
        # Note: BCx_t slices are separate channels; to access B and x_proj, we pass two separate tensors (B and x_proj) to this kernel.
        # However, we need to reconstruct B and x_proj from BCx_t. Simpler: host code creates B and x_proj slices and passes them here.
        # Since we have (B, S, H) slices, B_val is already from channel 0 and x_val from channel 1.
        bx = B_val * x_val
        tl.store(Bx_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, bx)


# Kernel: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: (B, S, H) contiguous input
# conv_weight: (H, H, 4) contiguous
# conv_bias: (H,) contiguous
# conv_out: (B, H, S) contiguous output
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_s, stride_bx_h,   # Bx strides
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_h, stride_out_s,  # conv_out strides
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
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_h + x_pos * stride_bx_s)  # note: x[b, ci, t+k] => h stride is stride_bx_h; t+k along s
            # conv_weight[ci, ci, k]
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_h + t * stride_out_s, acc)


# Kernel: Gating with C read from BCx_t: y = C * conv_out
# BCx_t: (B, 3H, S) view. C is channel 2H: shape (B, 1, S) logically. We pass C as (B, S, 1) and conv_out as (B, H, S).
@triton.jit
def gating_mul_y_kernel_BCx_t(
    BCx_ptr,        # *f32, shape (B, S, 1) derived from BCx_t[:, 2H:, :]
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H) output
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_h,  # BCx_t strides for channel 2H slice (actual h is 0); bc_h stride for last dim
    stride_conv_b, stride_conv_h, stride_conv_s,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    for h in range(0, H):
        C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_h)  # channel 2H, h stride is 0 since only 1 channel
        # conv_out[b, h, s]
        conv_val = tl.load(conv_out_ptr + b * stride_conv_b + h * stride_conv_h + s * stride_conv_s)
        y_val = C_val * conv_val
        tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H) input
# out_proj_weight: (H, hidden_size) contiguous
# out_proj_bias: (hidden_size,) contiguous
# output: (B, S, hidden_size)
@triton.jit
def linear_final_kernel(
    y_ptr,             # *f32, shape (B, S, H)
    out_proj_w_ptr,    # *f32, shape (H, hidden_size)
    out_proj_b_ptr,    # *f32, shape (hidden_size,)
    output_ptr,        # *f32, shape (B, S, hidden_size)
    B, S, H,           # ints (not used in kernel, but can be used if needed)
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_h, stride_w_oc,
    stride_out_b, stride_out_s, stride_out_oc,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    oc = tl.program_id(2)  # output channel index
    if b >= B or s >= S or oc < 0:
        return

    acc = 0.0
    for h in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h)
        w_val = tl.load(out_proj_w_ptr + h * stride_w_h + oc * stride_w_oc)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_b_ptr + oc)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + oc * stride_out_oc, acc)


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
        device = x.device

        # 1) Triple linear projection
        # Make inputs float32 and contiguous for F.linear
        x_f32 = x.contiguous().to(torch.float32)                      # (B, S, H)
        in_w_f32 = in_proj_weight.contiguous().to(torch.float32)     # (3H, H)
        in_b_f32 = in_proj_bias.contiguous().to(torch.float32)       # (3H,)

        BCx = torch.nn.functional.linear(x_f32, in_w_f32, in_b_f32)  # (B, S, 3H), float32

        # 2) Transpose to match reference layout exactly; do NOT call .contiguous() here.
        # This creates a non-contiguous view. The rest of the forward uses this view.
        BCx_t = BCx.transpose(-1, -2)  # (B, 3H, S) view

        # Prepare output tensors (we will pass slices to Triton kernels)
        B, S, H = x_f32.shape
        Nproj = in_w_f32.shape[0]  # 3 * H

        # 3) Reconstruct B and x_proj from BCx_t and compute Bx = B * x_proj
        # B slice: (B, S, H), channels 0..H-1
        B_slice = BCx_t[:, :H, :]
        # x_proj slice: (B, S, H), channels H..2H-1
        x_proj_slice = BCx_t[:, H:2 * H, :]

        # Ensure they are contiguous for Triton (explicitly make them contiguous)
        B_slice = B_slice.contiguous()   # (B, S, H)
        x_proj_slice = x_proj_slice.contiguous()  # (B, S, H)

        # Allocate Bx
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch gating_mul_kernel_BCx_t
        grid1 = (B, S)
        gating_mul_kernel_BCx_t[grid1](
            B_slice, x_proj_slice, Bx,
            B, S, H,
            B_slice.stride(0), B_slice.stride(1), B_slice.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Grouped causal conv1d: Bx -> conv_out
        # conv_out shape: (B, H, S), contiguous
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        # Ensure Bx is contiguous
        Bx_contig = Bx.contiguous()  # (B, S, H)

        conv_weight_f32 = conv_weight.contiguous().to(torch.float32)  # (H, H, 4)
        conv_bias_f32 = conv_bias.contiguous().to(torch.float32)      # (H,)

        grid2 = (B, H)
        causal_conv_groups_kernel[grid2](
            Bx_contig, conv_weight_f32, conv_bias_f32, conv_out,
            B, S, H,
            Bx_contig.stride(0), Bx_contig.stride(1), Bx_contig.stride(2),
            conv_weight_f32.stride(0), conv_weight_f32.stride(1), conv_weight_f32.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Gating with C: read C from BCx_t (channel 2H)
        C_slice = BCx_t[:, 2 * H:, :]  # shape (B, 1, S), view
        # To pass to Triton kernel, we need (B, S, 1). Create contiguous.
        C_contig = C_slice.transpose(1, 2).contiguous()  # (B, S, 1)

        y = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid3 = (B, S)
        gating_mul_y_kernel_BCx_t[grid3](
            C_contig, conv_out, y,
            B, S, H,
            C_contig.stride(0), C_contig.stride(1), C_contig.stride(2),  # BCx_t slice strides: b, s, h
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 6) Final linear projection
        out_proj_w_f32 = out_proj_weight.contiguous().to(torch.float32)  # (H, hidden_size)
        out_proj_b_f32 = out_proj_bias.contiguous().to(torch.float32)    # (hidden_size,)

        hidden_size = out_proj_w_f32.shape[1]
        output = torch.empty((B, S, hidden_size), device=device, dtype=torch.float32)

        grid5 = (B, S, hidden_size)
        linear_final_kernel[grid5](
            y, out_proj_w_f32, out_proj_b_f32, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w_f32.stride(0), out_proj_w_f32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
