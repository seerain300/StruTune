import torch
import triton
import triton.language as tl


# Kernel 1: triple_linear for B: output[b, s, ci] = sum_h x[b, s, h] * in_proj_w[ci, h] + bias[ci]
@triton.jit
def triple_linear_B_kernel(
    x_ptr,                 # *f32, shape (B, S, H)
    in_proj_w_ptr,         # *f32, shape (H, H)  # first H channels
    in_proj_b_ptr,         # *f32, shape (H,)
    B_out_ptr,             # *f32, shape (B, S, H)
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bo_b, stride_bo_s, stride_bo_ci,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = tl.program_id(2)  # output channel index in [0, H)

    if (b >= B) or (s >= S) or (ci >= H):
        return

    acc = 0.0
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_w_ptr + ci * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val
    b_bias = tl.load(in_proj_b_ptr + ci)
    acc += b_bias

    tl.store(B_out_ptr + b * stride_bo_b + s * stride_bo_s + ci * stride_bo_ci, acc)


# Kernel 2: triple_linear for x_proj: output[b, s, ci] = sum_h x[b, s, h] * in_proj_w[H + ci, h] + bias[H + ci]
@triton.jit
def triple_linear_xproj_kernel(
    x_ptr,                 # *f32, shape (B, S, H)
    in_proj_w_ptr,         # *f32, shape (H, H)  # middle H channels
    in_proj_b_ptr,         # *f32, shape (H,)
    x_proj_ptr,            # *f32, shape (B, S, H)
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_xp_b, stride_xp_s, stride_xp_ci,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = tl.program_id(2)  # output channel index in [0, H)

    if (b >= B) or (s >= S) or (ci >= H):
        return

    acc = 0.0
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_w_ptr + (H + ci) * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val
    b_bias = tl.load(in_proj_b_ptr + (H + ci))
    acc += b_bias

    tl.store(x_proj_ptr + b * stride_xp_b + s * stride_xp_s + ci * stride_xp_ci, acc)


# Kernel 3: triple_linear for C: output[b, s, ci] = sum_h x[b, s, h] * in_proj_w[2H + ci, h] + bias[2H + ci]
@triton.jit
def triple_linear_C_kernel(
    x_ptr,                 # *f32, shape (B, S, H)
    in_proj_w_ptr,         # *f32, shape (H, H)  # last H channels
    in_proj_b_ptr,         # *f32, shape (H,)
    C_ptr,                 # *f32, shape (B, S, H)
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_c_b, stride_c_s, stride_c_ci,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    ci = tl.program_id(2)  # output channel index in [0, H)

    if (b >= B) or (s >= S) or (ci >= H):
        return

    acc = 0.0
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_w_ptr + (2 * H + ci) * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val
    b_bias = tl.load(in_proj_b_ptr + (2 * H + ci))
    acc += b_bias

    tl.store(C_ptr + b * stride_c_b + s * stride_c_s + ci * stride_c_ci, acc)


# Kernel 4: Elementwise gating Bx = B * x_proj
@triton.jit
def gating_mul_kernel(
    B_ptr, x_ptr, Out_ptr,
    B, S, H,
    stride_b_b, stride_b_s, stride_b_h,
    stride_x_b, stride_x_s, stride_x_h,
    stride_out_b, stride_out_s, stride_out_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if (b >= B) or (s >= S) or (h >= H):
        return

    b_val = tl.load(B_ptr + b * stride_b_b + s * stride_b_s + h * stride_b_h)
    x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
    out_val = b_val * x_val
    tl.store(Out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, out_val)


# Kernel 5: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Inputs:
#   Bx: (B, S, H) conceptual indexing as (b, ci, t), where we map t = s, ci = h.
#   conv_weight: (H, H, 4), conv_bias: (H,)
# Outputs:
#   conv_out: (B, H, S) conceptual indexing as (b, ci, t).
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_w_go, stride_w_gi, stride_w_k,
    stride_co_b, stride_co_ci, stride_co_s,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output/input channel index
    if (b >= B) or (ci >= H) or (S <= 0):
        return

    acc = 0.0
    # Iterate over output time positions
    for t in range(0, S):
        # Causal conv: accumulate x[b, ci, t+k] * w[ci, ci, k] for k in 0..3
        for k in range(0, 4):
            x_pos = t + k
            if x_pos < S:
                x_val = tl.load(Bx_ptr + b * stride_bx_b + x_pos * stride_bx_s + ci * stride_bx_h)
                w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
                acc += x_val * w_val
        # Add bias
        bias_val = tl.load(conv_bias_ptr + ci)
        acc += bias_val
        # Store at (b, ci, t)
        tl.store(conv_out_ptr + b * stride_co_b + ci * stride_co_ci + t * stride_co_s, acc)
        acc = 0.0  # reset accumulator for next t


# Kernel 6: Elementwise gating y = C * conv_out
# Note: conv_out is (B, H, S). We pass conv_out as (B, S, H) and index accordingly.
@triton.jit
def gating_mul_y_kernel(
    C_ptr, conv_out_ptr, y_ptr,
    B, S, H,
    stride_c_b, stride_c_s, stride_c_h,
    stride_co_b, stride_co_s, stride_co_h,
    stride_y_b, stride_y_s, stride_y_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if (b >= B) or (s >= S) or (h >= H):
        return

    c_val = tl.load(C_ptr + b * stride_c_b + s * stride_c_s + h * stride_c_h)
    co_val = tl.load(conv_out_ptr + b * stride_co_b + s * stride_co_s + h * stride_co_h)
    out_val = c_val * co_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, out_val)


# Kernel 7: Final linear projection: output[b, s, h] = sum_h' y[b, s, h'] * out_proj_w[h, h'] + out_proj_b[h]
# Implemented as reduction over h' = H for each (b, s, h). We use a 3D grid over (b, s, h).
@triton.jit
def linear_final_kernel(
    y_ptr, out_proj_w_ptr, out_proj_b_ptr, out_ptr,
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_row, stride_w_col,  # out_proj_w shape (H, H): row = h, col = h'
    stride_out_b, stride_out_s, stride_out_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if (b >= B) or (s >= S) or (h >= H):
        return

    acc = 0.0
    # Reduce over h'
    for h2 in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h)  # y[b, s, h] is scalar per (b,s,h)
        # Note: we need y[b, s, h'], but since we vectorize across h, we keep y fixed; instead, we must load y[b, s, h2].
        # Correct: load y[b, s, h2] for each h2
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + h2 * stride_y_h)
        w_val = tl.load(out_proj_w_ptr + h * stride_w_row + h2 * stride_w_col)
        acc += y_val * w_val
    b_bias = tl.load(out_proj_b_ptr + h)
    acc += b_bias
    tl.store(out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous
        dtype = torch.float32
        B, S, H = x.shape
        x = x.contiguous().to(dtype)

        # Split in_proj_weight into three parts: H, H, H
        w1 = in_proj_weight[:H, :].contiguous().to(dtype)
        b1 = in_proj_bias[:H].contiguous().to(dtype)
        w2 = in_proj_weight[H:2*H, :].contiguous().to(dtype)
        b2 = in_proj_bias[H:2*H].contiguous().to(dtype)
        w3 = in_proj_weight[2*H:3*H, :].contiguous().to(dtype)
        b3 = in_proj_bias[2*H:3*H].contiguous().to(dtype)

        conv_w = conv_weight.contiguous().to(dtype)  # (H, H, 4)
        conv_b = conv_bias.contiguous().to(dtype)    # (H,)

        out_proj_w = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_proj_b = out_proj_bias.contiguous().to(dtype)    # (H,)

        # Allocate outputs
        B_t = torch.empty((B, S, H), device=x.device, dtype=dtype)
        x_proj = torch.empty((B, S, H), device=x.device, dtype=dtype)
        C = torch.empty((B, S, H), device=x.device, dtype=dtype)

        # Launch triple_linear kernels
        grid_tl = (B, S, H)
        triple_linear_B_kernel[grid_tl](
            x, w1, b1, B_t, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            w1.stride(0), w1.stride(1),
            B_t.stride(0), B_t.stride(1), B_t.stride(2),
            num_warps=1, num_stages=1,
        )

        triple_linear_xproj_kernel[grid_tl](
            x, w2, b2, x_proj, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            w2.stride(0), w2.stride(1),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            num_warps=1, num_stages=1,
        )

        triple_linear_C_kernel[grid_tl](
            x, w3, b3, C, B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            w3.stride(0), w3.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=1, num_stages=1,
        )

        # Gating Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=dtype)
        grid_gm = (B, S, H)
        gating_mul_kernel[grid_gm](
            B_t, x_proj, Bx,
            B, S, H,
            B_t.stride(0), B_t.stride(1), B_t.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # Grouped causal conv1d: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=dtype)
        grid_conv = (B, H)
        causal_conv_groups_kernel[grid_conv](
            Bx, conv_w, conv_b, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_w.stride(0), conv_w.stride(1), conv_w.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Gating y = C * conv_out
        # conv_out is (B, H, S); index as (b, s, h) by using conv_out[b, h, s]
        y = torch.empty((B, S, H), device=x.device, dtype=dtype)
        grid_gmy = (B, S, H)
        gating_mul_y_kernel[grid_gmy](
            C, conv_out, y,
            B, S, H,
            C.stride(0), C.stride(1), C.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # Final linear projection: output = F.linear(y, out_proj_w, out_proj_b)
        # Implement as batched dot over H for each (b, s, h)
        output = torch.empty((B, S, H), device=x.device, dtype=dtype)
        grid_fl = (B, S, H)
        linear_final_kernel[grid_fl](
            y, out_proj_w, out_proj_b, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w.stride(0), out_proj_w.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
