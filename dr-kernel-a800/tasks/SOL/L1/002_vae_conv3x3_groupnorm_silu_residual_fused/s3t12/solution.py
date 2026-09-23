import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H, W). Each program computes exactly one output element.
@triton.jit
def conv3x3_triton_single(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    h = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = 0.0
    # loop over input channels
    for ic in range(0, C_in):
        # accumulate over 3x3 neighborhood with padding=1
        for kh in range(-1, 2):
            h_in = h + kh
            valid_h = (h_in >= 0) & (h_in < H)
            for kw in range(-1, 2):
                w_in = w_out + kw
                valid_w = (w_in >= 0) & (w_in < W)
                in_bounds = valid_h & valid_w

                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + h_in * x_stride_h \
                         + w_in * x_stride_w
                # If out-of-bounds due to padding, masked load sets 0
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)

                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_val = tl.load(w_ptrs)
                acc += x_val * w_val

    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w_out * y_stride_w
    tl.store(y_ptrs, acc)


# Triton kernel: elementwise residual add y = y + x (flattened)
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Args:
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: First conv weights (C, C, 3, 3)
        norm1_weight: First GroupNorm scale (C,)
        norm1_bias: First GroupNorm bias (C,)
        conv2_weight: Second conv weights (C, C, 3, 3)
        norm2_weight: Second GroupNorm scale (C,)
        norm2_bias: Second GroupNorm bias (C,)
        eps: Epsilon for GroupNorm numerical stability
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert x.dim() == 4, "Input must be NCHW"
    B, C, H, W = x.shape
    Cw1 = conv1_weight.shape[0]
    assert Cw1 == C, "conv1_weight C_in must match x channels C"
    Cw2 = conv2_weight.shape[0]
    assert Cw2 == C, "conv2_weight C_in must match x channels C"
    assert C % 32 == 0, "num_groups=32 requires C divisible by 32"

    # Allocate outputs
    out1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    out2 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

    # Launch conv1 (Triton)
    grid1 = (B, C, H, W)
    conv3x3_triton_single[grid1](
        x, conv1_weight, out1,
        B, C, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        num_warps=1, num_stages=1,
    )

    # GroupNorm + SiLU (PyTorch for correctness)
    out1 = F.group_norm(out1, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out1 = F.silu(out1)

    # Launch conv2 (Triton)
    grid2 = (B, C, H, W)
    conv3x3_triton_single[grid2](
        out1, conv2_weight, out2,
        B, C, C, H, W,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        num_warps=1, num_stages=1,
    )

    # GroupNorm + SiLU (PyTorch)
    out2 = F.group_norm(out2, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out2 = F.silu(out2)

    # Residual add via Triton (final output is out2 + x)
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 4096),)
    final = torch.empty_like(out2)
    add_residual_kernel[grid_add](
        final, out2, x, N, 4096
    )

    return final


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
