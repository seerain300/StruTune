import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H) float32
# in_proj_weight: (Nproj, H) float32, Nproj = 3 * H
# in_proj_bias: (Nproj,) float32
# BCx_out: (B, S, Nproj) float32
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
    b = tl.load(in_proj_bias_ptr + co)
    acc += b  # add bias (note: here 'b' is bias, not the x_val)
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating: Bx = B * x_proj
# BCx: (B, S, 3H) with B at co=0, x_proj at co=1 (conceptually), we read via BCx[:, :, 0] and [:, :, 1]
# Bx_out: (B, H, S) but we write to Bx_out using (b, s, h) addressing via strides.
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                 # *f32, shape (B, S, Nproj=3H)
    Bx_ptr,                  # *f32, shape (B, H, S)
    B, S, H,                 # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_h, stride_bx_s,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return

    # Read B at co=0 and x_proj at co=1 from BCx, then multiply
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    res = B_val * x_val
    tl.store(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s * stride_bx_s, res)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx_in: (B, H, S) conceptual indexing as (b, ci, t); here we pass Bx as (B, S, H) but we'll read (b, ci, t) via strides.
# conv_weight: (H, H, 4) float32, conv_bias: (H,) float32
# conv_out: (B, H, S_out) where S_out = S - 3 (causal padding on left)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H) but accessed as (b, ci, t)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S_out)
    B, S, H, S_out,     # ints
    stride_bx_b, stride_bx_ci, stride_bx_t,   # for Bx logical (b, ci, t)
    stride_w_go, stride_w_gi, stride_w_k,     # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,# for conv_out (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S_out <= 0:
        return

    for t in range(0, S_out):
        acc = 0.0
        # Causal conv with kernel_size=4: k in {0,1,2,3}
        for k in range(0, 4):
            x_pos = t + k
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val
        bias_val = tl.load(conv_bias_ptr + ci)
        acc += bias_val
        tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating: y = C * conv_out; C is the third chunk of BCx: (B, S, H) conceptual, we access as (b, ci=2, t)
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,                # *f32, shape (B, S, Nproj=3H)
    conv_out_ptr,           # *f32, shape (B, H, S_out)
    y_ptr,                  # *f32, shape (B, H, S_out)
    B, S, H, S_out,         # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_out_b, stride_out_ci, stride_out_t,
    stride_y_b, stride_y_h, stride_y_s,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    if b >= B or t >= S_out or h >= H:
        return

    C_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_ci + t * stride_out_t)
    res = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + h * stride_y_h + t * stride_y_s, res)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S_out, H) conceptual; we write output[b, s, h] = sum_h y[b, s, h] * out_proj_weight[h, h] + out_proj_bias[h]
# Note: out_proj_weight is (H, H), but we only use the diagonal to match F.linear(y, W, b) where y has H out features.
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S_out, H) but we index (b, s, h) via strides
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    out_ptr,                # *f32, shape (B, S_out, H)
    B, S_out, H,            # ints
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_row, stride_w_col,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S_out or h >= H:
        return

    # Accumulate over H dimension: output[b, s, h] = sum_ci y[b, s, ci] * W[ci, h] + bias[h]
    acc = 0.0
    for ci in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + ci * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + ci * stride_w_row + h * stride_w_col)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val
    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure all tensors are float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        Nproj = 3 * H  # triple projection

        # 1) Triple linear: BCx (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), device=x.device, dtype=torch.float32)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj
        # We need to index BCx[:, :, 0] and [:, :, 1] as (b, s, co). For Triton, we use strides and co=0 and co=1.
        Bx = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid2 = (B, H, S)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv (kernel_size=4, groups=H) on Bx
        # Conv expects (N, C_in, L). We pass Bx as (B, H, S) logically and compute conv_out (B, H, S-3)
        S_out = S - 3  # causal padding implies output length equals input length minus kernel_size + 1
        conv_out = torch.empty((B, H, S_out), device=x.device, dtype=torch.float32)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H, S_out,
            Bx.stride(0), 1, 1,               # logical strides for (b, ci, t): we pass strides that Triton uses to index logically
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), 1,  # conv_out strides: (b, ci, t)
            num_warps=1, num_stages=1,
        )

        # 4) Gating: y = C * conv_out
        y = torch.empty((B, H, S_out), device=x.device, dtype=torch.float32)
        grid4 = (B, H, S_out)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H, S_out,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), 1,
            y.stride(0), y.stride(1), 1,
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection to output (B, S_out, H)
        # Note: out_proj_weight is (H, H); we implement F.linear(y, W, b) with diagonal weights.
        output = torch.empty((B, S_out, H), device=x.device, dtype=torch.float32)
        grid5 = (B, S_out, H)
        linear_final_kernel[grid5](
            y, out_proj_weight, out_proj_bias, output,
            B, S_out, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
