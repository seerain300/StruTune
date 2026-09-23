import torch
import triton
import triton.language as tl


# Triton kernel for elementwise gating: Bx = B * x_proj
# BCx is (B, S, 3H); we read B from channel 0 and x_proj from channel 1, then write Bx as (B, S, H).
@triton.jit
def gating_mul_kernel(
    BCx_ptr,                # *f32, shape (B, S, Nproj), Nproj >= 3H
    Bx_ptr,                 # *f32, shape (B, S, H)
    B, S, H, Nproj,         # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_bx_b, stride_bx_s, stride_bx_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)  # output channel index in [0, H), corresponds to x_proj channel index 1
    if b >= B or s >= S or h >= H:
        return

    # Read B = BCx[b, s, 0], x_proj = BCx[b, s, 1]
    B_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj_val = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = B_val * x_proj_val
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Host-side helper to run gating_mul_kernel
def gating_mul_forward(BCx):
    B, S, Nproj = BCx.shape
    H = Nproj // 3
    Bx = torch.empty((B, S, H), device=BCx.device, dtype=torch.float32)

    BCx_c = BCx.contiguous().to(torch.float32)
    Bx_c = Bx

    grid = (B, S, H)
    gating_mul_kernel[grid](
        BCx_c, Bx_c,
        B, S, H, Nproj,
        BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
        Bx_c.stride(0), Bx_c.stride(1), Bx_c.stride(2),
        num_warps=2, num_stages=1,
    )
    return Bx_c


# Triton kernel for y = C * conv_out, reading C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, shape (B, S, Nproj)
    conv_out_ptr,   # *f32, shape (B, H, S)
    y_ptr,          # *f32, shape (B, H, S)
    B, S, H, Nproj, # ints
    stride_bc_b, stride_bc_s, stride_bc_co,
    stride_out_b, stride_out_ci, stride_out_t,
    stride_y_b, stride_y_ci, stride_y_t,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index
    t = tl.program_id(2)   # time index
    if b >= B or ci >= H or t >= S:
        return

    # Read C = BCx[b, s, 2] for s = t
    C_val = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 2 * stride_bc_co)
    y_val = tl.load(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t)
    y_val = C_val * y_val
    tl.store(y_ptr + b * stride_y_b + ci * stride_y_ci + t * stride_y_t, y_val)


# Host-side helper to run gating_mul_y_kernel
def gating_mul_y_forward(BCx, conv_out):
    B, S, Nproj = BCx.shape
    B_out, H, S_out = conv_out.shape
    assert B_out == B and S_out == S, "Shape mismatch between BCx and conv_out for gating"
    y = torch.empty((B, H, S), device=BCx.device, dtype=torch.float32)

    BCx_c = BCx.contiguous().to(torch.float32)
    out_c = conv_out.contiguous().to(torch.float32)
    y_c = y  # float32

    grid = (B, H, S)
    gating_mul_y_kernel[grid](
        BCx_c, out_c, y_c,
        B, S, H, Nproj,
        BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
        out_c.stride(0), out_c.stride(1), out_c.stride(2),
        y_c.stride(0), y_c.stride(1), y_c.stride(2),
        num_warps=2, num_stages=1,
    )
    return y_c


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Step 1: Triple linear projection via PyTorch (fast and robust)
        # x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,)
        BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Step 2: Element-wise gating via Triton: Bx = B * x_proj
        Bx = gating_mul_forward(BCx)  # (B, S, H)

        # Step 3: Grouped causal 1D convolution via PyTorch (avoid long Triton loops)
        # Input for conv: (B, H, S) conceptual mapping. F.pad adds causal left padding.
        pad = conv_weight.shape[2] - 1  # kernel_size
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))
        # conv_weight: (H, H, kernel_size), conv_bias: (H,), groups=H (depthwise)
        conv_out = torch.nn.functional.conv1d(Bx_padded, conv_weight, conv_bias, groups=H)  # (B, H, S)

        # Step 4: Output gating via Triton: y = C * conv_out, reading C from BCx[:, 2, :]
        y = gating_mul_y_forward(BCx, conv_out)  # (B, H, S)

        # Step 5: Final output projection via PyTorch: output = linear(y_T, out_proj_weight, out_proj_bias)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)
        output = torch.nn.functional.linear(y_T, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
