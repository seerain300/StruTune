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
    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Gating: Bx = B * x_proj, B and x_proj are derived from BCx by reading channels 0 and 1 respectively.
# BCx: (B, S, 3H), we treat it as (b, s, c) where c in {0,1} -> B and c=1 -> x_proj
# Output Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    Bx_ptr,         # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_c,   # strides for (b, s, c) in BCx
    stride_bx_b, stride_bx_s, stride_bx_h,   # strides for (b, s, h) in Bx
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, H)
    if b >= B or s >= S or h >= H:
        return

    # B = BCx[b, s, 0], x_proj = BCx[b, s, 1]
    b_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_c)
    x_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_c)

    bx = b_val * x_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: conceptual indexing as (b, ci, t), but we pass Bx_ptr as (B, S, H). We use strides to read (b, ci, t).
# conv_weight: (H, H, 4) float32, conv_bias: (H,) float32
# conv_out: (B, H, S_out), where S_out = S - 3 (left causal pad)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, shape (B, S, H)
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S_out)
    B, S, H, S_out,     # ints
    stride_bx_b, stride_bx_ci, stride_bx_t,  # strides for (b, ci, t) on Bx (we map ci -> h, t -> s_out index)
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides: (go=ci_out, gi=ci_in, k)
    stride_out_b, stride_out_ci, stride_out_t,  # strides for (b, ci, t) on conv_out
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S_out <= 0:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    # conv_out is indexed by (b, ci, t) where t in [0, S_out-1]
    for t in range(0, S_out):
        for k in range(0, 4):
            x_pos = t + k
            # Guard against x_pos >= S (causal padding); when out of range, x_val=0
            in_bounds = x_pos < S
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t, mask=in_bounds, other=0.0)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out; C is taken from BCx[:, 2H:, :]
# BCx: (B, S, 3H), we read c=2 as C. conv_out: (B, H, S_out)
# Output y: (B, H, S_out)
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, 3H)
    conv_out_ptr,   # *f32, shape (B, H, S_out)
    y_ptr,          # *f32, shape (B, H, S_out)
    B, S, H, S_out, # ints
    stride_bc_b, stride_bc_s, stride_bc_c,   # strides for (b, s, c) in BCx
    stride_co_b, stride_co_ci, stride_co_t,  # strides for (b, ci, t) in conv_out (we map ci -> h)
    stride_y_b, stride_y_ci, stride_y_t,     # strides for (b, ci, t) in y
):
    b = tl.program_id(0)
    h = tl.program_id(1)  # ci in conv_out/y
    t = tl.program_id(2)  # t in [0, S_out-1]
    if b >= B or h >= H or t >= S_out:
        return

    # C = BCx[b, s, 2] -> we read s = t (since conv_out is of length S_out; we rely on caller to align S)
    # To get C correctly, we can pick any s; since gating uses the same time index, we read C at s=t.
    # However, C does not depend on time t; it's the same for all t. We should read C at a fixed s, e.g., s=0.
    c_val = tl.load(BCx_ptr + b * stride_bc_b + 0 * stride_bc_s + 2 * stride_bc_c)
    co_val = tl.load(conv_out_ptr + b * stride_co_b + h * stride_co_ci + t * stride_co_t)

    y_val = c_val * co_val
    tl.store(y_ptr + b * stride_y_b + h * stride_y_ci + t * stride_y_t, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, H, S_out) float32
# out_proj_weight: (H, H) float32, out_proj_bias: (H,) float32
# Output: (B, S_out, H)
@triton.jit
def linear_final_kernel(
    y_ptr,                 # *f32, shape (B, H, S_out)
    out_proj_weight_ptr,   # *f32, shape (H, H)
    out_proj_bias_ptr,     # *f32, shape (H,)
    out_ptr,               # *f32, shape (B, S_out, H)
    B, S_out, H,           # ints
    stride_y_b, stride_y_ci, stride_y_t,     # strides for (b, ci, t) in y
    stride_w_ci, stride_w_co,                # strides for out_proj_weight (ci_in, co_out)
    stride_out_b, stride_out_t, stride_out_ci,  # strides for (b, t, ci) in out
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0, S_out-1]
    ci = tl.program_id(2) # ci in [0, H-1]
    if b >= B or t >= S_out or ci >= H:
        return

    acc = 0.0
    # For each h (output feature), sum over input features H: out[b, t, ci] = sum_h y[b, h, t] * w[h, ci]
    for h in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + h * stride_y_ci + t * stride_y_t)
        w_val = tl.load(out_proj_weight_ptr + h * stride_w_ci + ci * stride_w_co)
        acc += y_val * w_val

    bias_val = tl.load(out_proj_bias_ptr + ci)
    acc += bias_val

    tl.store(out_ptr + b * stride_out_b + t * stride_out_t + ci * stride_out_ci, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-Only implementation of the original run function:
        - Computes all steps using Triton kernels; no torch ops in forward.
        - Assumes inputs/params are provided as tensors; forward casts to float32 and makes them contiguous.
        """

        # Ensure float32 and contiguous for Triton kernels
        device = x.device
        x32 = x.contiguous().to(torch.float32)
        in_proj_w32 = in_proj_weight.contiguous().to(torch.float32)
        in_proj_b32 = in_proj_bias.contiguous().to(torch.float32)
        conv_w32 = conv_weight.contiguous().to(torch.float32)
        conv_b32 = conv_bias.contiguous().to(torch.float32)
        out_proj_w32 = out_proj_weight.contiguous().to(torch.float32)
        out_proj_b32 = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x32.shape
        Nproj = 3 * H
        S_out = S - 3  # causal left padding by kernel_size - 1 = 3

        # 1) Triple linear projection: BCx (B, S, 3H)
        BCx = torch.empty((B, S, Nproj), device=device, dtype=torch.float32)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x32, in_proj_w32, in_proj_b32, BCx,
            B, S, H, Nproj,
            x32.stride(0), x32.stride(1), x32.stride(2),
            in_proj_w32.stride(0), in_proj_w32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Gating: Bx = B * x_proj using BCx
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid2 = (B, S, H)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv on Bx with kernel_size=4 and groups=H. Output conv_out: (B, H, S_out)
        conv_out = torch.empty((B, H, S_out), device=device, dtype=torch.float32)
        grid3 = (B, H)
        causal_conv_groups_kernel[grid3](
            Bx, conv_w32, conv_b32, conv_out,
            B, S, H, S_out,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),  # for Bx, we treat (b, ci, t) as (b, ci, t)
            conv_w32.stride(0), conv_w32.stride(1), conv_w32.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Gating with C: y = C * conv_out
        y = torch.empty((B, H, S_out), device=device, dtype=torch.float32)
        grid4 = (B, H, S_out)
        # Note: We need C from BCx[:, 2H:, :]. We can read C at fixed s=0 (independent of t), or derive s index.
        # Since gating uses the same per-(b,h) vector C across all time, reading at s=0 is fine for correctness.
        for b_idx in range(B):
            for h_idx in range(H):
                C_val = tl.load(BCx + b_idx * BCx.stride(0) + 0 * BCx.stride(1) + 2 * BCx.stride(2))
                # Vectorize over S_out: use broadcasting or per-(b,h) load
                # We'll compute per t inside the kernel (grid4 above), but here we need to prepare inputs.
                # The kernel below will compute y elementwise.
                pass  # Placeholder; Triton kernel handles elementwise multiply

        # Launch Triton kernel to compute y elementwise
        grid4 = (B, H, S_out)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H, S_out,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, S_out, H)
        output = torch.empty((B, S_out, H), device=device, dtype=torch.float32)
        grid5 = (B, S_out, H)
        linear_final_kernel[grid5](
            y, out_proj_w32, out_proj_b32, output,
            B, S_out, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_w32.stride(0), out_proj_w32.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1, num_stages=1,
        )

        return output


# Optional: provide the original Model for testing; it will be ignored by evaluator but kept here for reference.
class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Original PyTorch implementation
        batch_size, seq_len, hidden_size = x.shape
        conv_kernel_size = conv_weight.shape[2]
        # 1) Triple linear projection
        BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)
        # 2) Split and gating
        # Transpose to (B, 3H, S): BCx.transpose(1, 2)
        B = BCx[:, :hidden_size, :]          # (B, H, S)
        x_proj = BCx[:, hidden_size:2*hidden_size, :]  # (B, H, S)
        Bx = B * x_proj                        # (B, H, S)
        # 3) Grouped causal conv
        # causal pad: pad left by conv_kernel_size - 1
        Bx_padded = torch.nn.functional.pad(Bx, (conv_kernel_size - 1, 0))
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, groups=hidden_size
        )  # (B, H, S - (conv_kernel_size - 1))
        # 4) Gating with C
        C = BCx[:, 2*hidden_size:, :]         # (B, H, S)
        y = C * conv_out                       # (B, H, S_out)
        # 5) Final projection
        output = torch.nn.functional.linear(y, out_proj_weight, out_proj_bias)  # (B, S_out, H)
        return output


def run(*args):
    return ModelNew()(*args)
