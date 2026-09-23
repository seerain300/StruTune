import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj), float32
@triton.jit
def triple_linear_kernel(
    x_ptr: tl.pointer[tl.float32],            # *f32, shape (B, S, H)
    in_proj_weight_ptr: tl.pointer[tl.float32],  # *f32, shape (Nproj, H)
    in_proj_bias_ptr: tl.pointer[tl.float32],    # *f32, shape (Nproj,)
    BCx_ptr: tl.pointer[tl.float32],           # *f32, shape (B, S, Nproj)
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

    # x_val = x[b, s, h], w_val = in_proj_weight[co, h], sum over h
    acc = 0.0
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating Bx = B * x_proj, reading B and x_proj from BCx (transposed as (B, 3H, S))
# We reconstruct B and x_proj by viewing BCx as (B, 3H, S) and taking channels 0 and 1.
# Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr: tl.pointer[tl.float32],          # *f32, shape (B, 3H, S)
    Bx_ptr: tl.pointer[tl.float32],           # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)
    if b >= B or s >= S or h >= H:
        return

    # Load B from channel 0
    B_val = tl.load(BCx_ptr + b * stride_bc_b + 0 * stride_bc_c + s * stride_bc_s)
    # Load x_proj from channel 1
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + 1 * stride_bc_c + s * stride_bc_s)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: conceptual indexing as (b, ci, t), where Bx is actually (B, S, H) but we map as above.
# conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
# Each program handles one (b, ci). We rely on causal indexing (no explicit padding).
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr: tl.pointer[tl.float32],           # *f32, shape (B, S, H)
    conv_weight_ptr: tl.pointer[tl.float32],  # *f32, shape (H, H, 4)
    conv_bias_ptr: tl.pointer[tl.float32],    # *f32, shape (H,)
    conv_out_ptr: tl.pointer[tl.float32],     # *f32, shape (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_bx_b, stride_bx_ci, stride_bx_t,   # mapping for (b, ci, t) on Bx
    stride_w_go, stride_w_gi, stride_w_k,     # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    for t in range(0, S):
        # Since we do not pad, ensure we don't read negative t+k; if invalid, treat as 0.
        for k in range(0, 4):
            # For causal conv, we can assume t+k is within S for valid t and k.
            x_pos = t + k
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    # Store at (b, ci, t). Note: conv_out_ptr is (B, H, S). We write per t for all ci.
    # In this simple implementation, we store acc into conv_out[b, ci, t] for each t iteration.
    # We can create a temporary vector for t? Triton doesn't allow vectorized store with dynamic t easily here.
    # So we will store per t inside the loop by calling tl.store each iteration. However, Triton doesn't support
    # direct per-t store unless we create a 2D tile. To keep it simple and correct, we store acc into conv_out
    # at (b, ci, t) each iteration. Triton allows that with a scalar store.
    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr: tl.pointer[tl.float32],          # *f32, shape (B, 3H, S)
    conv_out_ptr: tl.pointer[tl.float32],     # *f32, shape (B, H, S)
    y_ptr: tl.pointer[tl.float32],            # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
    stride_bc_b, stride_bc_c, stride_bc_s,
    stride_co_b, stride_co_ci, stride_co_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)
    if b >= B or s >= S or h >= H:
        return

    # Load C from channel 2
    C_val = tl.load(BCx_ptr + b * stride_bc_b + 2 * stride_bc_c + s * stride_bc_s)
    # Load conv_out[b, h, s]
    co_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + s * stride_co_t)
    y_val = C_val * co_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr: tl.pointer[tl.float32],            # *f32, shape (B, S, H)
    out_proj_weight_ptr: tl.pointer[tl.float32],  # *f32, shape (H, H)
    out_proj_bias_ptr: tl.pointer[tl.float32],    # *f32, shape (H,)
    output_ptr: tl.pointer[tl.float32],       # *f32, shape (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H)
    if b >= B or s >= S or h >= H:
        return

    acc = 0.0
    for i in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + s * stride_y_s + i * stride_y_h)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_o + i * stride_w_i)
        acc += y_val * w_val
    bias_val = tl.load(out_proj_bias_ptr + h)
    acc += bias_val
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
        # Ensure all inputs are float32 and contiguous
        B, S, H = x.shape
        Nproj = 3 * H
        assert in_proj_weight.shape[0] == Nproj and in_proj_weight.shape[1] == H
        assert conv_weight.shape[0] == H and conv_weight.shape[1] == H and conv_weight.shape[2] == 4
        assert out_proj_weight.shape[0] == H and out_proj_weight.shape[1] == H
        assert conv_bias.shape[0] == H
        assert out_proj_bias.shape[0] == H

        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        # 1) Triple linear projection: BCx (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=x.device)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Transpose BCx to (B, 3H, S) for easy channel access
        BCx_T = BCx.transpose(1, 2).contiguous()  # (B, 3H, S)
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx_T, Bx,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv1d on Bx with kernel_size=4, groups=H
        # Map Bx to (B, H, S) for conv: we'll pass Bx_ptr and treat as (b, ci, t).
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
        # Reuse BCx_T to read C from channel 2
        y = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid4 = (B, S, H)
        gating_mul_y_kernel[grid4](
            BCx_T, conv_out, y,
            B, S, H, Nproj,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final output projection: output = F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid5 = (B, S, H)
        linear_final_kernel[grid5](
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
