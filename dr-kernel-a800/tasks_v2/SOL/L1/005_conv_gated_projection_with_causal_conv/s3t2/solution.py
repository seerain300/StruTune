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
    H_BLOCK: tl.constexpr,  # block size for H vectorization
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    # Vectorized reduction over H dimension
    acc = 0.0
    for ci in range(0, H, H_BLOCK):
        idx = ci + tl.arange(0, H_BLOCK)
        mask = idx < H
        x_vec = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + idx * stride_x_h, mask=mask, other=0.0)
        w_vec = tl.load(in_proj_weight_ptr + co * stride_w_co + idx * stride_w_ci, mask=mask, other=0.0)
        # dot-product accumulate
        acc += tl.sum(x_vec * w_vec, axis=0)
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating from BCx: reconstruct B and x_proj and compute Bx = B * x_proj
# Bx: (B, S, H)
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
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    # channel 0 -> B, channel 1 -> x_proj
    b_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = b_vec * x_proj
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx logical shape: (B, S, H). We pass Bx as (B, H, S) via strides for conv.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, H, S) but we access via strides; logical (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_go, stride_w_gi, stride_w_k,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)  # output channel index in [0, H)
    for t in range(0, S):
        acc = 0.0
        for k in range(0, 4):
            t_in = t - (k - 1)  # causal padding: index t - (k - 1)
            in_mask = (t_in >= 0) and (t_in < S)
            val = tl.load(Bx_ptr + b * stride_bx_b + oc * stride_bx_h + t_in * stride_bx_s, mask=in_mask, other=0.0)
            w = tl.load(conv_weight_ptr + oc * stride_w_go + oc * stride_w_gi + k * stride_w_k)
            acc += val * w
        bias = tl.load(conv_bias_ptr + oc)
        acc += bias
        tl.store(conv_out_ptr + b * stride_bx_b + oc * stride_bx_h + t * stride_bx_s, acc)


# Kernel 4: Output gating: y = C * conv_out
# C comes from BCx[:, 2, :], conv_out: (B, H, S)
@triton.jit
def gating_mul_kernel_y(
    BCx_ptr,         # *f32, shape (B, S, 3H), channel 2 is C
    conv_out_ptr,    # *f32, shape (B, H, S)
    y_ptr,           # *f32, shape (B, H, S)
    B, S, H,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_co_b, stride_co_h, stride_co_s,
    stride_y_b, stride_y_h, stride_y_s,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    c = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    co = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_h + s * stride_co_s)
    y = c * co
    tl.store(y_ptr + b * stride_y_b + h * stride_y_h + s * stride_y_s, y)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,               # *f32, shape (B, S, H)
    out_proj_weight_ptr, # *f32, shape (H, H)
    out_proj_bias_ptr,   # *f32, shape (H,)
    output_ptr,          # *f32, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for i in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + i * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_o + i * stride_w_i)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + h)
    acc += bias
    tl.store(output_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


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
        # Cast to float32 for Triton kernels and ensure contiguity
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]
        assert Nproj == 3 * H, "in_proj_weight first dim must be 3*H"
        assert conv_weight.shape == (H, H, 4), "conv_weight must be (H, H, 4)"
        assert conv_bias.shape == (H,), "conv_bias must be (H,)"
        assert out_proj_weight.shape == (H, H), "out_proj_weight must be (H, H)"
        assert out_proj_bias.shape == (H,), "out_proj_bias must be (H,)"

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, Nproj), device=x.device, dtype=torch.float32)

        # Vectorized over H: choose a reasonable block size (e.g., 128). If H is small, it still works.
        H_BLOCK = 128
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            H_BLOCK=H_BLOCK,
            num_warps=2, num_stages=1,
        )

        # 2) Element-wise gating: Bx = B * x_proj, where B = BCx[:, :, 0], x_proj = BCx[:, :, 1]
        BCx_c = BCx.contiguous()
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_c, Bx,
            B, S, H,
            BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=2, num_stages=1,
        )

        # 3) Grouped causal 1D convolution: conv_out = conv(Bx, conv_weight, conv_bias), groups=H
        # We pass Bx as (B, H, S) via strides for the kernel.
        Bx_bhs = Bx.permute(0, 2, 1).contiguous()  # (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx_bhs, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx_bhs.stride(0), Bx_bhs.stride(1), Bx_bhs.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            num_warps=2, num_stages=1,
        )

        # 4) Output gating: y = C * conv_out, where C = BCx[:, :, 2*H:] (shape (B, S, H))
        C = BCx[:, :, 2 * H:].contiguous()  # (B, S, H)
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        grid4 = (B, H, S)
        gating_mul_kernel_y[grid4](
            BCx_c, conv_out, y,
            B, S, H,
            BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=2, num_stages=1,
        )

        # 5) Final linear projection: output = F.linear(y, out_proj_weight, out_proj_bias)
        y_bsh = y.permute(0, 2, 1).contiguous()  # (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y_bsh, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y_bsh.stride(0), y_bsh.stride(1), y_bsh.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=2, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
