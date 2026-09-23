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
    # One program per output element: (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            ih = oh - pad_h + kh
            iw = ow - pad_w + kw

            # Check input bounds
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute input pointer
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # Compute weight pointer (no bias)
            w_off = c * stride_wC + 0 * stride_wKH + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_off)

            acc += x_val * w_val

    # Store output
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


def triton_depthwise_conv2d_groupsC(x: torch.Tensor, weight: torch.Tensor, padding: int = 3):
    """
    Triton implementation of depthwise conv2d with groups=C.
    x: (B, C, H, W), float32
    weight: (C, 1, 7, 7), float32
    padding: int, symmetric padding on H and W
    returns y: (B, C, H + 2*padding, W + 2*padding), float32
    """
    assert x.ndim == 4 and weight.ndim == 4, "x must be (B,C,H,W), weight must be (C,1,7,7)"
    B, C, H, W = x.shape
    # Ensure tensors are on CUDA
    x_c = x
    w_c = weight
    # Output shape
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Allocate output
    y = torch.empty((B, C, H_out, W_out), device=x.device, dtype=torch.float32)

    # Use explicit strides
    stride_xB, stride_xC, stride_xH, stride_xW = x_c.stride()
    stride_wC, stride_wKH, stride_wKW = w_c.stride()  # 1x7x7 so wKH and wKW are strides for 7x7
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_output_kernel[grid](
        x_c, w_c, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=4,  # reasonable default for this kernel
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized model: compute depthwise conv2d (groups=C) using Triton,
    and return the conv output as the main result.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor, device: torch.device = None, **kwargs):
        # If not on CUDA, move to current CUDA device for Triton
        if residual.device.type != "cuda":
            residual = residual.to(torch.device("cuda"))
        if dwconv_weight.device.type != "cuda":
            dwconv_weight = dwconv_weight.to(torch.device("cuda"))

        # Compute depthwise conv2d with groups=C using Triton
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)

        # Return the Triton-computed result (match expected signature)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
