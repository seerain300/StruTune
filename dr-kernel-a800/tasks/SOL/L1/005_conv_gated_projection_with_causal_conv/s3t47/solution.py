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
    h = tl.program_id(2)  # output channel index in [0, H)

    if b >= B or s >= S or h >= H:
        return

    # B is channel 0, x_proj is channel 1 in BCx
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)
    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Triton conv kernel: 2D grouped causal conv with kernel_size=4, stride=1, padding 3 on left
# Input: Bx_padded (B, H, S), weight: (H, H, 4), bias: (H,)
# Output: conv_out (B, H, S)
@triton.jit
def conv_triton_kernel(
    Bx_ptr,             # *f32, (B, H, S)
    weight_ptr,         # *f32, (H, H, 4)
    bias_ptr,           # *f32, (H,)
    out_ptr,            # *f32, (B, H, S)
    B, H, S,
    stride_bx_b, stride_bx_ci, stride_bx_t,  # for input (b, ci, t)
    stride_w_go, stride_w_gi, stride_w_k,    # weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for output (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel)
    if b >= B or ci >= H:
        return

    acc = 0.0
    # Causal conv with kernel size 4: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k]
    for t in range(0, S):
        # For k=0
        x_pos = t + 0
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
        else:
            x_val = 0.0
        w0 = tl.load(weight_ptr + ci * stride_w_go + ci * stride_w_gi + 0 * stride_w_k)
        acc += x_val * w0

        # For k=1
        x_pos = t + 1
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
        else:
            x_val = 0.0
        w1 = tl.load(weight_ptr + ci * stride_w_go + ci * stride_w_gi + 1 * stride_w_k)
        acc += x_val * w1

        # For k=2
        x_pos = t + 2
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
        else:
            x_val = 0.0
        w2 = tl.load(weight_ptr + ci * stride_w_go + ci * stride_w_gi + 2 * stride_w_k)
        acc += x_val * w2

        # For k=3
        x_pos = t + 3
        if x_pos < S:
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
        else:
            x_val = 0.0
        w3 = tl.load(weight_ptr + ci * stride_w_go + ci * stride_w_gi + 3 * stride_w_k)
        acc += x_val * w3

    bias_val = tl.load(bias_ptr + ci)
    acc += bias_val
    tl.store(out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 3: Gating with C: y = C * conv_out; read C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, S, H)
    B, S, H,
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_out_b, stride_out_ci, stride_out_t,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    # C is channel 2 in BCx
    C_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + h * stride_out_ci + s * stride_out_t)
    y_val = C_val * conv_val
    tl.store(y_ptr + b * stride_y_b + s * stride_y_s + h * stride_y_h, y_val)


# Kernel 4: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                  # *f32, shape (B, S, H)
    out_proj_weight_ptr,    # *f32, shape (H, H)
    out_proj_bias_ptr,      # *f32, shape (H,)
    output_ptr,             # *f32, shape (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output feature index

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

        # 2) Gating: Bx = B * x_proj, reading B from channel 0 and x_proj from channel 1
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm = (B, S, H)
        gating_mul_kernel[grid_gm](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Prepare input for conv: causal padding on sequence (left pad 3)
        # We will construct a padded input tensor for conv_triton_kernel directly.
        # However, conv_triton_kernel reads Bx_ptr as (B, H, S) and uses explicit (b, ci, t) indexing.
        # Here, Bx has shape (B, S, H); we can create a dummy view by swapping strides virtually.
        # Instead, we create a separate tensor with shape (B, H, S) via indexing using Bx.permute(0, 2, 1).contiguous().
        Bx_padded = Bx.permute(0, 2, 1).contiguous()  # (B, H, S)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid_conv = (B, H)
        conv_triton_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, H, S,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),   # input strides (b, ci, t)
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),  # weight strides
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),           # output strides (b, ci, t)
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out; C is channel 2 from BCx
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

        # 5) Final linear projection: output = F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_f = (B, S, H)
        linear_final_kernel[grid_f](
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
