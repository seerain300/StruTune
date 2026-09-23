import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), float32
@triton.jit
def triple_linear_kernel(
    x_ptr: tl.pointer[tl.float32],                  # *f32, shape (B, S, H)
    in_proj_weight_ptr: tl.pointer[tl.float32],     # *f32, shape (Nproj, H)
    in_proj_bias_ptr: tl.pointer[tl.float32],       # *f32, shape (Nproj,)
    BCx_ptr: tl.pointer[tl.float32],                # *f32, shape (B, S, Nproj)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
    stride_x_b: tl.constexpr, stride_x_s: tl.constexpr, stride_x_h: tl.constexpr,
    stride_w_co: tl.constexpr, stride_w_ci: tl.constexpr,
    stride_bc_b: tl.constexpr, stride_bc_s: tl.constexpr, stride_bc_co: tl.constexpr,
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

    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating: split BCx -> B and x_proj and compute Bx = B * x_proj
# BCx: (B, S, 3H), Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr: tl.pointer[tl.float32],        # *f32, shape (B, S, 3H)
    Bx_ptr: tl.pointer[tl.float32],         # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b: tl.constexpr, stride_bc_s: tl.constexpr, stride_bc_co: tl.constexpr,
    stride_bx_b: tl.constexpr, stride_bx_s: tl.constexpr, stride_bx_h: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index within H
    if b >= B or s >= S or h >= H:
        return

    # Channel 0 is B, channel 1 is x_proj
    b_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = b_vec * x_proj
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: conceptual (B, H, S) using strides to index (b, ci, t)
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr: tl.pointer[tl.float32],             # *f32, shape (B, S, H)
    conv_weight_ptr: tl.pointer[tl.float32],    # *f32, shape (H, H, 4)
    conv_bias_ptr: tl.pointer[tl.float32],      # *f32, shape (H,)
    conv_out_ptr: tl.pointer[tl.float32],       # *f32, shape (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bx_b: tl.constexpr, stride_bx_ci: tl.constexpr, stride_bx_t: tl.constexpr,  # mapping for (b, ci, t) on Bx
    stride_w_go: tl.constexpr, stride_w_gi: tl.constexpr, stride_w_k: tl.constexpr,    # conv_weight strides
    stride_out_b: tl.constexpr, stride_out_ci: tl.constexpr, stride_out_t: tl.constexpr,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S <= 0:
        return

    acc = 0.0

    # y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr: tl.pointer[tl.float32],        # *f32, shape (B, S, 3H)
    conv_out_ptr: tl.pointer[tl.float32],   # *f32, shape (B, H, S)
    y_ptr: tl.pointer[tl.float32],          # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bc_b: tl.constexpr, stride_bc_s: tl.constexpr, stride_bc_co: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_ci: tl.constexpr, stride_out_t: tl.constexpr,
    stride_y_b: tl.constexpr, stride_y_s: tl.constexpr, stride_y_h: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    out_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_ci + s * stride_out_t)
    y_val = C_val * out_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H) stored as float32
@triton.jit
def linear_final_kernel(
    y_ptr: tl.pointer[tl.float32],                   # *f32, shape (B, S, H)
    out_proj_weight_ptr: tl.pointer[tl.float32],     # *f32, shape (H, H)
    out_proj_bias_ptr: tl.pointer[tl.float32],       # *f32, shape (H,)
    out_ptr: tl.pointer[tl.float32],                 # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_y_b: tl.constexpr, stride_y_s: tl.constexpr, stride_y_h: tl.constexpr,
    stride_w_go: tl.constexpr, stride_w_gi: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_s: tl.constexpr, stride_out_h: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for go in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + go * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_go + go * stride_w_gi)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + h)
    acc += bias

    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor, conv_weight: torch.Tensor,
                conv_bias: torch.Tensor, out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        # Cast all inputs/params to float32 to avoid dtype issues in Triton
        x_f32 = x.to(torch.float32).contiguous()
        in_proj_weight_f32 = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias_f32 = in_proj_bias.to(torch.float32).contiguous()
        conv_weight_f32 = conv_weight.to(torch.float32).contiguous()
        conv_bias_f32 = conv_bias.to(torch.float32).contiguous()
        out_proj_weight_f32 = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias_f32 = out_proj_bias.to(torch.float32).contiguous()

        B, S, H = x_f32.shape
        Nproj = in_proj_weight_f32.shape[0]  # 3 * H
        assert Nproj == 3 * H

        # 1) Triple linear projection BCx: (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=x_f32.device)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x_f32, in_proj_weight_f32, in_proj_bias_f32, BCx,
            B, S, H, Nproj,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            in_proj_weight_f32.stride(0), in_proj_weight_f32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: compute Bx = B * x_proj, where B=BCx[:, 0, :], x_proj=BCx[:, 1, :]
        # We need Bx shape (B, S, H)
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x_f32.device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv: Bx -> (B, H, S) conceptual, conv_weight (H, H, 4)
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x_f32.device)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight_f32, conv_bias_f32, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # (b, ci, t)
            conv_weight_f32.stride(0), conv_weight_f32.stride(1), conv_weight_f32.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),  # (b, ci, t)
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out; C is BCx[:, 2, :]
        y = torch.empty((B, S, H), dtype=torch.float32, device=x_f32.device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection: y -> out
        out = torch.empty((B, S, H), dtype=torch.float32, device=x_f32.device)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y, out_proj_weight_f32, out_proj_bias_f32, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight_f32.stride(0), out_proj_weight_f32.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Return out (float32)
        return out


def run(*args):
    return ModelNew()(*args)
