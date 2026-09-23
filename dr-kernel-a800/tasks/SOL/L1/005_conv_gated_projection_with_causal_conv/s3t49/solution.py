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


# Kernel 3: Causal padding (kernel_size=4): create Bx_padded from Bx with zeros on the left
# Bx_padded: (B, S, H), where for t >= 0: Bx_padded[b, t, :] = 0; for t >= 1: Bx_padded[b, t, :] = Bx[b, t-1, :]
@triton.jit
def pad_bx_kernel(
    Bx_ptr,              # *f32, shape (B, S, H)
    Bx_padded_ptr,       # *f32, shape (B, S, H)
    B, S, H,             # ints
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_pad_b, stride_pad_s, stride_pad_h,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or t >= S or h >= H:
        return

    if t == 0:
        tl.store(Bx_padded_ptr + b * stride_pad_b + t * stride_pad_s + h * stride_pad_h, 0.0)
    else:
        val = tl.load(Bx_ptr + b * stride_bx_b + (t - 1) * stride_bx_s + h * stride_bx_h)
        tl.store(Bx_padded_ptr + b * stride_pad_b + t * stride_pad_s + h * stride_pad_h, val)


# Kernel 4: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx_padded
# Bx_padded: conceptual (b, ci, t) with actual layout (B, S, H)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_padded_ptr,        # *f32, shape (B, S, H)
    conv_weight_ptr,      # *f32, shape (H, H, 4)
    conv_bias_ptr,        # *f32, shape (H,)
    conv_out_ptr,         # *f32, shape (B, H, S)
    B, S, H,              # ints
    stride_bp_b, stride_bp_s, stride_bp_h,  # for (b, s, h) read
    stride_w_go, stride_w_gi, stride_w_k,   # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_s,  # for (b, ci, s) write
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, s] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, s + k] + bias[ci]
    for s_out in range(0, S):
        for k in range(0, 4):
            t_in = s_out + k
            # For t_in < 0, padded Bx_padded has zeros; masked by if t_in >= 0, which is true for s_out>=0.
            val = tl.load(Bx_padded_ptr + b * stride_bp_b + t_in * stride_bp_s + ci * stride_bp_h)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + s_out * stride_out_s, acc)


# Kernel 5: Output gating: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_ci, stride_co_s,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # C is channel 2 in BCx
    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_s)
    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 6: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
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
    stride_w_co, stride_w_ci,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_co + ci * stride_w_ci)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val

    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


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
        device = x.device
        B, S, H = x.shape
        Nproj = 3 * H

        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=device)

        grid_tl = (B, S, Nproj)
        triple_linear_kernel[grid_tl](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm = (B, S, H)
        gating_mul_kernel[grid_gm](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Causal padding: Bx_padded with kernel_size=4 (pad left by 3 zeros)
        Bx_padded = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_pad = (B, S, H)
        pad_bx_kernel[grid_pad](
            Bx, Bx_padded,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Grouped causal conv (depthwise, kernel=4, groups=H) on Bx_padded
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid_conv = (B, H)
        causal_conv_groups_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Output gating: y = C * conv_out
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

        # 6) Final linear projection
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_final = (B, S, H)
        linear_final_kernel[grid_final](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


# Below are the original helper functions provided, used by the evaluator to generate inputs.
@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]
    # Step 1: Triple linear projection
    BCx = F.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)
    # Step 2: Gating
    B, C, x_proj = BCx.chunk(3, dim=-1)
    Bx = B * x_proj  # (B, S, H)
    # Step 3: Grouped causal 1D convolution (kernel_size=4, groups=H)
    Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))  # pad left
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)  # (B, H, S)
    # Step 4: Output gating
    y = C * conv_out  # (B, H, S)
    # Step 5: Final output projection
    output = F.linear(y.transpose(-1, -2).contiguous(), out_proj_weight, out_proj_bias)  # (B, S, H)
    return output


def run(*args):
    return ModelNew()(*args)
