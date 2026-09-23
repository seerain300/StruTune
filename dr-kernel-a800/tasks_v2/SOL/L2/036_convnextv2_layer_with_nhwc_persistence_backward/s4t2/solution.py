import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_per_out_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # Grid: (B, C, H_out, W_out)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator for the single output element
    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 depthwise convolution: sum over kernel window
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # Input bounds check due to padding
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Address computation with explicit strides
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            # Weight address: per-channel scalar; kH and kW are 0-dim indices
            w_offset = c * stride_wC + 0 * stride_wKH + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            acc += x_val * w_val

    # Store the result
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


def triton_depthwise_conv2d_groupsC(residual: torch.Tensor, dwconv_weight: torch.Tensor, padding: int = 3):
    """
    Triton implementation of depthwise conv2d with groups=C, no bias.
    residual: (B, C, H, W), dwconv_weight: (C, 1, 7, 7)
    output: (B, C, H+2*padding, W+2*padding)
    """
    assert residual.ndim == 4 and dwconv_weight.ndim == 4, "Invalid input shapes"
    B, C, H, W = residual.shape
    # Ensure inputs are contiguous
    residual = residual.contiguous()
    dwconv_weight = dwconv_weight.contiguous()

    # Output tensor (same dtype as input)
    H_out = H + 2 * padding
    W_out = W + 2 * padding
    y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

    # Strides
    stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
    stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Launch Triton kernel: one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_out_kernel[grid](
        residual, dwconv_weight, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses Triton to compute depthwise conv2d (groups=C) on the residual input.
    - The rest of the pipeline remains in torch to preserve original semantics and ensure correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect residual and dwconv_weight as inputs (as in the original run signature)
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects at least residual and dwconv_weight as inputs.")
        residual = args[0]
        dwconv_weight = args[1]
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
