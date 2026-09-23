import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection BCx[b, s, co] = sum_h in_proj_weight[co, h] * x[b, s, h] + in_proj_bias[co]
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output BCx: (B, S, Nproj)
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, shape (B, S, H)
    in_proj_weight_ptr,     # *f32, shape (Nproj, H)
    in_proj_bias_ptr,       # *f32, shape (Nproj,)
    BCx_ptr,                # *f32, shape (B, S, Nproj)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
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
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating Bx = B * x_proj
# BCx is (B, S, 3H); we reconstruct B (co=0) and x_proj (co=1) and compute Bx = B * x_proj
@triton.jit
def gating_mul_kernel(
    BCx_ptr,            # *f32, shape (B, S, 3H)
    Bx_ptr,             # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = 0  # channel index 0 corresponds to 'B'
    if b >= B or s >= S or ci >= H:
        return
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    ci_val = 1  # channel index 1 corresponds to 'x_proj'
    if ci_val >= 3 * H:
        return
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + ci_val * stride_bc_co)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + ci * stride_bx_co, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: conceptual mapping (b, ci, t) where Bx is (B, S, H)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S) conceptual
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bx_b, stride_bx_ci, stride_bx_t,
    stride_w_go, stride_w_gi, stride_w_k,
    stride_out_b, stride_out_ci, stride_out_t,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output and input channel index (groups=H)
    if b >= B or ci >= H or S <= 0:
        return

    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            # Safe load with mask: if x_pos >= S, skip
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
                w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
                acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_ci,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = 0  # output channel index (also input channel, groups=H)
    if b >= B or s >= S or ci >= H:
        return

    # Load C from BCx[:, 2, :]
    c_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)

    # Load conv_out[b, ci, s]
    co_val = tl.load(conv_out_ptr + b * stride_co_b + ci * stride_co_ci + s * stride_co_t)

    y_val = c_val * co_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_ci, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, hidden_size), out_proj_bias: (hidden_size,)
# output: (B, S, hidden_size)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, hidden_size)
    out_proj_bias_ptr,      # *f32, shape (hidden_size,)
    out_ptr,                # *f32, shape (B, S, hidden_size)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, hidden_size: tl.constexpr,
    stride_y_b, stride_y_s, stride_y_ci,
    stride_w_go, stride_w_gi,
    stride_out_b, stride_out_s, stride_out_ci,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = tl.program_id(2)  # output feature index [0, hidden_size)
    if b >= B or s >= S or ci >= hidden_size:
        return

    acc = 0.0
    for h in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_ci)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_go + ci * stride_w_gi)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + ci)
    acc += bias
    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + ci * stride_out_ci, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        1) BCx = F.linear(x, in_proj_weight, in_proj_bias) -> shape (B, S, 3H)
        2) Bx = B * x_proj (B = BCx[:, :, 0], x_proj = BCx[:, :, 1])
        3) conv_out = grouped causal conv1d(Bx, kernel=4, groups=H)
        4) y = C * conv_out (C = BCx[:, :, 2])
        5) output = F.linear(y, out_proj_weight, out_proj_bias)
        """
        # Ensure float32 and contiguous
        device = x.device
        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]  # 3 * H

        x_f32 = x.contiguous().to(torch.float32)
        in_proj_w_f32 = in_proj_weight.contiguous().to(torch.float32)
        in_proj_b_f32 = in_proj_bias.contiguous().to(torch.float32)

        # 1) Triple linear projection
        BCx = torch.empty((B, S, Nproj), device=device, dtype=torch.float32)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x_f32, in_proj_w_f32, in_proj_b_f32, BCx,
            B, S, H, Nproj,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            in_proj_w_f32.stride(0), in_proj_w_f32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv1d on Bx (kernel_size=4), groups=H
        conv_w_f32 = conv_weight.contiguous().to(torch.float32)  # (H, H, 4)
        conv_b_f32 = conv_bias.contiguous().to(torch.float32)    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # conceptual (B, H, S)

        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_w_f32, conv_b_f32, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # map (b, ci, t): stride_bx_b, stride_bx_ci, stride_bx_t
            conv_w_f32.stride(0), conv_w_f32.stride(1), conv_w_f32.stride(2),
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
        out_proj_w_f32 = out_proj_weight.contiguous().to(torch.float32)  # (H, hidden_size)
        out_proj_b_f32 = out_proj_bias.contiguous().to(torch.float32)    # (hidden_size,)
        hidden_size = out_proj_w_f32.shape[1]
        output = torch.empty((B, S, hidden_size), device=device, dtype=torch.float32)

        grid5 = (B, S, hidden_size)
        linear_final_kernel[grid5](
            y, out_proj_w_f32, out_proj_b_f32, output,
            B, S, H, hidden_size,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w_f32.stride(0), out_proj_w_f32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
