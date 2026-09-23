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


# Kernel 2: Element-wise gating: reconstruct B and x_proj from BCx last dim and compute Bx = B * x_proj
# BCx: (B, S, 3H), Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    Bx_ptr,         # *f32, shape (B, S, H)
    B, S, H,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    b_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = b_vec * x_proj
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# We vectorize over output channels in tiles of BLOCK_OC. Logical input Bx is (B, S, H).
# Bx is provided as (B, H, S) via strides to the kernel for indexing convenience.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel_vec(
    Bx_ptr,             # *f32, shape (B, H, S) but we access via strides; logical (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_go, stride_w_gi, stride_w_k,
    BLOCK_OC: tl.constexpr,
):
    b = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    # Vector of output channels in this tile
    oc_vec = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_vec < H

    # For each output time index t
    for t in range(0, S):
        acc_vec = tl.zeros([BLOCK_OC], dtype=tl.float32)
        # Reduce over kernel taps k=0..3
        for k in range(0, 4):
            t_in = t - (k - 1)  # causal padding: shift by (k-1)
            in_mask = (t_in >= 0) & (t_in < S)
            # For each oc in this tile, accumulate conv input
            for j in range(0, BLOCK_OC):
                oc = oc_vec[j]
                if oc_mask[j]:
                    val = 0.0
                    if in_mask:
                        # logical Bx[b, t_in, oc] via strides: (B, H, S)
                        val = tl.load(Bx_ptr + b * stride_bx_b + oc * stride_bx_h + t_in * stride_bx_s)
                    w = tl.load(conv_weight_ptr + oc * stride_w_go + oc * stride_w_gi + k * stride_w_k)
                    acc_vec[j] += val * w
        # Add bias
        bias_vec = tl.load(conv_bias_ptr + oc_vec, mask=oc_mask, other=0.0)
        acc_vec += bias_vec
        # Store conv_out[b, oc_vec, t]
        tl.store(conv_out_ptr + b * stride_bx_b + oc_vec * stride_bx_h + t * stride_bx_s, acc_vec, mask=oc_mask)


# Kernel 4: Output gating: y = C * conv_out, C comes from BCx channel 2
# BCx: (B, S, 3H) -> C = BCx[:, :, 2*H:] shape (B, S, H)
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

    c = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + (2 * H) * stride_bc_co)
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
        # Shapes
        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]
        assert Nproj == 3 * H, "in_proj_weight first dim must be 3*H"
        assert conv_weight.shape == (H, H, 4), "conv_weight must be (H, H, 4)"
        assert conv_bias.shape == (H,), "conv_bias must be (H,)"
        assert out_proj_weight.shape == (H, H), "out_proj_weight must be (H, H)"
        assert out_proj_bias.shape == (H,), "out_proj_bias must be (H,)"

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        x_c = x.contiguous()
        in_proj_w_c = in_proj_weight.contiguous()
        in_proj_b_c = in_proj_bias.contiguous()

        BCx = torch.empty((B, S, Nproj), device=x.device, dtype=x.dtype)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x_c, in_proj_w_c, in_proj_b_c, BCx,
            B, S, H, Nproj,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            in_proj_w_c.stride(0), in_proj_w_c.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Element-wise gating: Bx = B * x_proj, where B = BCx[:, :, 0], x_proj = BCx[:, :, 1]
        BCx_c = BCx.contiguous()
        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_c, Bx,
            B, S, H,
            BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D convolution: conv_out = conv(Bx, conv_weight, conv_bias), groups=H
        # We pass Bx as (B, H, S) via strides for the kernel.
        Bx_bhs = Bx.permute(0, 2, 1).contiguous()  # (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        conv_w_c = conv_weight.contiguous()  # (H, H, 4)
        conv_b_c = conv_bias.contiguous()

        BLOCK_OC = 32
        grid3 = (B, _ceil_div(H, BLOCK_OC))
        causal_conv_groups_kernel_vec[grid3](
            Bx_bhs, conv_w_c, conv_b_c, conv_out,
            B, S, H,
            Bx_bhs.stride(0), Bx_bhs.stride(1), Bx_bhs.stride(2),
            conv_w_c.stride(0), conv_w_c.stride(1), conv_w_c.stride(2),
            BLOCK_OC=BLOCK_OC,
            num_warps=1, num_stages=1,
        )

        # 4) Output gating: y = C * conv_out, where C = BCx[:, :, 2*H:] (shape (B, S, H))
        C = BCx[:, :, 2 * H:].contiguous()  # (B, S, H)
        y = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        grid4 = (B, H, S)
        gating_mul_kernel_y[grid4](
            BCx_c, conv_out, y,
            B, S, H,
            BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection: output = F.linear(y, out_proj_weight, out_proj_bias)
        y_bsh = y.permute(0, 2, 1).contiguous()  # (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        out_proj_w_c = out_proj_weight.contiguous()
        out_proj_b_c = out_proj_bias.contiguous()

        grid5 = (B, S, H)
        linear_final_kernel[grid5](
            y_bsh, out_proj_w_c, out_proj_b_c, output,
            B, S, H,
            y_bsh.stride(0), y_bsh.stride(1), y_bsh.stride(2),
            out_proj_w_c.stride(0), out_proj_w_c.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
