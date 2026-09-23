import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # One program computes one output element y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = 0.0

    # Iterate over 7x7 kernel taps
    for kh in range(7):
        ih = oh + kh - pad
        in_h = (ih >= 0) & (ih < H)
        for kw in range(7):
            iw = ow + kw - pad
            in_w = (iw >= 0) & (iw < W)
            in_bounds = in_h & in_w

            # Compute input pointer: x[b, c, ih, iw]
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Masked load (safe even if in_bounds is False)
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Compute weight pointer: w[c, 0, kh, kw]
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            # Accumulate
            acc += x_val * w_val

    # Store result to y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses Triton to compute depthwise conv2d (groups=C) on the input.
    - The rest of the pipeline is in torch to preserve original semantics and ensure correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual, dwconv_weight):
        """
        residual: (B, C, H, W)
        dwconv_weight: (C, 1, 7, 7)
        Returns: y: (B, C, H+6, W+6) with padding=3
        """
        # Ensure CUDA and contiguous
        if not residual.is_cuda:
            residual = residual.cuda(non_blocking=True)
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.cuda(non_blocking=True)
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        H_out = H + 6
        W_out = W + 6

        # Allocate output
        y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

        # Get strides
        stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            3,  # padding on height and width
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4,  # modest parallelism
            num_stages=3,
        )

        return y


def run(*args):
    return ModelNew()(*args)
