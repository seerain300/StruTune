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
    # Reduce over H
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val

    # Add bias
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


# Kernel: Pad Bx along sequence dimension (causal padding for kernel_size=4)
# Bx: (B, S, H) input
# Bx_padded: (B, S + 3, H) output (padding 3 zeros on the left)
@triton.jit
def pad_bx_kernel(
    Bx_ptr,            # *f32, input (B, S, H)
    Bx_padded_ptr,     # *f32, output (B, S+3, H)
    B, S, H,           # ints
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_p_b, stride_p_s, stride_p_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    # Out index starts at 3, so map s_out = s + 3
    s_out = s + 3

    val = tl.load(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h)
    tl.store(Bx_padded_ptr + b * stride_p_b + s_out * stride_p_s + h * stride_p_h, val)


# Kernel: Grouped causal 1D convolution (depthwise, kernel_size=4), groups=H
# Bx_padded: (B, S+3, H), conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, S, H)
@triton.jit
def conv1d_groups_kernel(
    Bx_padded_ptr,     # *f32, input (B, S+3, H) but we index as (b, ci, t)
    conv_weight_ptr,   # *f32, (H, H, 4)
    conv_bias_ptr,     # *f32, (H,)
    conv_out_ptr,      # *f32, output (B, S, H)
    B, S, H,           # ints
    stride_bx_b, stride_bx_ci, stride_bx_t,  # we'll set these by mapping: b, ci, t_out
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_t, stride_out_ci,  # for (b, t_out, ci)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output/input channel index, groups=H
    if b >= B or ci >= H:
        return

    # Accumulator
    acc = 0.0

    # Causal conv: y[b, ci, t_out] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t_out + k] + bias[ci]
    for t_out in range(0, S):
        for k in range(0, 4):
            t_in = t_out + k
            if t_in < S + 3:
                x_val = tl.load(Bx_padded_ptr + b * stride_bx_b + ci * stride_bx_ci + t_in * stride_bx_t)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store to conv_out at (b, t_out, ci)
    tl.store(conv_out_ptr + b * stride_out_b + t_out * stride_out_t + ci * stride_out_ci, acc)


# Kernel: y = C * conv_out, read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, S, H)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_s, stride_co_h,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # C is channel 2 in BCx
    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    y_val = C_val * tl.load(conv_out_ptr + b * stride_co_b + s * stride_co_s + h * stride_co_h)
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
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
    stride_w_oh, stride_w_oi,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)  # output channel index

    if b >= B or s >= S or h_out >= H:
        return

    acc = 0.0
    # Reduce over input H
    for h_in in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h_in * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_oh + h_in * stride_w_oi)
        acc += y_val * w_val

    # Add bias
    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,                      # (B, S, H)
        in_proj_weight: torch.Tensor,         # (3H, H)
        in_proj_bias: torch.Tensor,           # (3H,)
        conv_weight: torch.Tensor,            # (H, H, 4)
        conv_bias: torch.Tensor,              # (H,)
        out_proj_weight: torch.Tensor,        # (H, H)
        out_proj_bias: torch.Tensor,          # (H,)
    ):
        # Ensure float32 and contiguous
        B, S, H = x.shape
        device = x.device

        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, 3 * H), dtype=torch.float32, device=device)

        Nproj = 3 * H
        grid_tl = (B, S, Nproj)
        triple_linear_kernel[grid_tl](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Elementwise gating: Bx = B * x_proj
        # Reconstruct B and x_proj from BCx channels 0 and 1
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm = (B, S, H)
        gating_mul_kernel[grid_gm](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Causal padding along sequence (kernel_size=4 => pad 3 zeros on left)
        Bx_padded = torch.empty((B, S + 3, H), dtype=torch.float32, device=device)
        grid_pad = (B, S, H)
        pad_bx_kernel[grid_pad](
            Bx, Bx_padded,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Grouped causal conv1d: y = conv(Bx_padded, conv_weight, groups=H)
        conv_out = torch.empty((B, S, H), dtype=torch.float32, device=device)

        grid_conv = (B, H)
        conv1d_groups_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, S, H,
            # Map Bx_padded indexing as (b, ci, t): strides
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),  # (b, ci, t) -> ci uses stride(1), t uses stride(2)
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(2), conv_out.stride(1),     # store as (b, t, ci)
            num_warps=1, num_stages=1,
        )

        # 5) Output gating: y = C * conv_out (read C from BCx[:, 2, :])
        y = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gmy = (B, S, H)
        gating_mul_y_kernel[grid_gmy](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 6) Final output projection: F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_fin = (B, S, H)
        linear_final_kernel[grid_fin](
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
