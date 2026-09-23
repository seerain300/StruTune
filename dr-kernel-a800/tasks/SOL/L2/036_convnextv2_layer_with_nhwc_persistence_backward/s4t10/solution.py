import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # Each program computes one output y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel taps
    for kh in range(7):
        for kw in range(7):
            # Compute input coordinates with padding
            ih = oh - pad_h + kh
            iw = ow - pad_w + kw

            # Valid if within [0, H) and [0, W)
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Load input element x[b, c, ih, iw] if in-bounds, else 0
            # Address: b*stride_xB + c*stride_xC + ih*stride_xH + iw*stride_xW
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # Load weight for this channel and tap: w[c, 0, kh, kw]
            # Address: c*stride_wC + 0*stride_wKH + kh*stride_wKW + kw*stride_wKW
            w_off = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_off)

            # Accumulate
            acc += x_val * w_val

    # Store the result to y[b, c, oh, ow]
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


def triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3):
    """
    Compute depthwise conv2d with groups=C using Triton.
    Input residual: (B, C, H, W)
    Weight dwconv_weight: (C, 1, 7, 7)
    Output: y (B, C, H+2*padding, W+2*padding)
    """
    assert residual.is_cuda and dwconv_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    B, C, H, W = residual.shape
    # Ensure contiguous
    residual = residual.contiguous()
    dwconv_weight = dwconv_weight.contiguous()

    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Allocate output
    y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

    # Extract strides (in elements)
    stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
    stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Grid: one program per output element
    grid = (B, C, H_out, W_out)

    depthwise_conv2d_groupsC_per_output_kernel[grid](
        residual, dwconv_weight, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=1,  # simple kernel; 1 warp per program is sufficient
        num_stages=1,
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses Triton to compute depthwise conv2d (groups=C) on the residual input.
    - The rest of the pipeline is done in torch to preserve original semantics and ensure correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual, dwconv_weight):
        # Ensure tensors are on CUDA for Triton; if not, move to current device
        if not residual.is_cuda:
            residual = residual.to(torch.device("cuda"))
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.to(torch.device("cuda"))

        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
