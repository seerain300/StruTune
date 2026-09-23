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
    # One program per output element y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = 0.0

    # Loop over 7x7 kernel (depthwise conv, groups=C)
    for kh in range(7):
        for kw in range(7):
            # Compute input indices with padding
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # Mask for valid input indices
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Load input x[b, c, ih, iw] with zero when out-of-bounds
            x_addr = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_addr, mask=in_bounds, other=0.0)

            # Load weight w[c, 0, kh, kw]
            w_addr = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_addr)

            # Accumulate
            acc += x_val * w_val

    # Store result to y[b, c, oh, ow]
    y_addr = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_addr, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version that computes depthwise conv2d (groups=C) using Triton.
    The forward returns the conv output tensor y (B, C, H+6, W+6).
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA for Triton execution
        if residual.device.type != 'cuda':
            residual = residual.to('cuda', non_blocking=True)
        if dwconv_weight.device.type != 'cuda':
            dwconv_weight = dwconv_weight.to('cuda', non_blocking=True)

        B, C, H, W = residual.shape
        # Output size for depthwise conv with padding=3
        H_out = H + 2 * 3
        W_out = W + 2 * 3

        # Allocate output tensor (float32 for numeric stability)
        y = torch.empty((B, C, H_out, W_out), device='cuda', dtype=torch.float32)

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            3, 3,  # padding
            residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
            dwconv_weight.stride(0), dwconv_weight.stride(1), dwconv_weight.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=1, num_stages=1,
        )

        return y


def run(*args):
    return ModelNew()(*args)
