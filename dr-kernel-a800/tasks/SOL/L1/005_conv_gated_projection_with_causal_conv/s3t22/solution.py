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
    B, S, H, Nproj,         # ints (runtime)
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


# Kernel 2: Gating: Bx = B * x_proj
# We read B (channel 0) and x_proj (channel 1) from a permuted BCx of shape (B, 3H, S).
@triton.jit
def gating_mul_kernel(
    BCx_perm_ptr,   # *f32, shape (B, 3H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_co, stride_bc_s,  # strides for BCx_perm (B, 3H, S)
    stride_y_b, stride_y_s, stride_y_h,      # strides for y (B, S, H)
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    # Load B and x_proj from BCx_perm (channels 0 and 1), then multiply
    B_val = tl.load(BCx_perm_ptr + b * stride_bc_b + 0 * stride_bc_co + s * stride_bc_s)
    x_proj_val = tl.load(BCx_perm_ptr + b * stride_bc_b + 1 * stride_bc_co + s * stride_bc_s)
    bx = B_val * x_proj_val

    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx conceptual (B, H, S), but we pass y of shape (B, S, H). We map (b, ci, t) to y[b, t, ci].
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    y_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,   # *f32, shape (H, H, 4)
    conv_bias_ptr,     # *f32, shape (H,)
    conv_out_ptr,      # *f32, shape (B, H, S)
    B, S, H,           # ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_go, stride_w_gi, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # input/output channel index (groups=H)
    if b >= B or ci >= H:
        return

    acc = 0.0
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            if x_pos < S:
                x_val = tl.load(y_ptr + b * stride_y_b + x_pos * stride_y_s + ci * stride_y_h)
            else:
                x_val = 0.0
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store at (b, ci, t)
    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_h + t * stride_out_s, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_kernel_y(
    BCx_perm_ptr,   # *f32, shape (B, 3H, S)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,
    stride_bc_b, stride_bc_co, stride_bc_s,  # for BCx_perm (B, 3H, S)
    stride_out_b, stride_out_h, stride_out_s,  # for conv_out (B, H, S)
    stride_y_b, stride_y_s, stride_y_h,      # for y (B, S, H)
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    # Load C from BCx_perm channel 2 at (b, s, 2)
    C_val = tl.load(BCx_perm_ptr + b * stride_bc_b + 2 * stride_bc_co + s * stride_bc_s)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s)
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
    out_ptr,                # *f32, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    if b >= B or s >= S or h_out >= H:
        return

    acc = 0.0
    for h in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h_out * stride_w_o + h * stride_w_i)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h_out * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        device = x.device
        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]

        # Cast to float32 and ensure contiguity for Triton
        x32 = x.to(torch.float32).contiguous()
        in_proj_weight32 = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias32 = (in_proj_bias if in_proj_bias is not None else torch.zeros(Nproj, device=device, dtype=torch.float32)).contiguous()
        conv_weight32 = conv_weight.to(torch.float32).contiguous()
        conv_bias32 = (conv_bias if conv_bias is not None else torch.zeros(H, device=device, dtype=torch.float32)).contiguous()
        out_proj_weight32 = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias32 = (out_proj_bias if out_proj_bias is not None else torch.zeros(H, device=device, dtype=torch.float32)).contiguous()

        # 1) Triple linear projection: BCx (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), device=device, dtype=torch.float32)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x32, in_proj_weight32, in_proj_bias32, BCx,
            B, S, H, Nproj,
            x32.stride(0), x32.stride(1), x32.stride(2),
            in_proj_weight32.stride(0), in_proj_weight32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        BCx_perm = BCx.permute(0, 2, 1).contiguous()  # (B, 3H, S)
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_perm, Bx,
            B, S, H,
            BCx_perm.stride(0), BCx_perm.stride(1), BCx_perm.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight32, conv_bias32, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight32.stride(0), conv_weight32.stride(1), conv_weight32.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
        y_perm = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid4 = (B, S, H)
        gating_mul_kernel_y[grid4](
            BCx_perm, conv_out, y_perm,
            B, S, H,
            BCx_perm.stride(0), BCx_perm.stride(1), BCx_perm.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y_perm.stride(0), y_perm.stride(1), y_perm.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y_perm, out_proj_weight32, out_proj_bias32, output,
            B, S, H,
            y_perm.stride(0), y_perm.stride(1), y_perm.stride(2),
            out_proj_weight32.stride(0), out_proj_weight32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
