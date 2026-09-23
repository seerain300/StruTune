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
    # Program ids: one program per output element y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            # Compute input coordinates with padding
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # Mask input loads to valid range
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Load input: x[b, c, ih, iw]
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            x_val = x_val.to(tl.float32)

            # Load weight: w[c, 0, kh, kw]
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            w_val = w_val.to(tl.float32)

            acc += x_val * w_val

    # Store output: y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized model:
    - Uses Triton to compute depthwise conv2d with groups=C.
    - Keeps the rest of the pipeline in PyTorch to maximize correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual, dwconv_weight):
        # Ensure tensors are on CUDA and contiguous
        assert residual.is_cuda and dwconv_weight.is_cuda, "Inputs must be on CUDA for Triton."
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        # Depthwise conv with groups=C, kH=kW=7, padding=3
        pad = 3
        H_out = H + 2 * pad
        W_out = W + 2 * pad

        # Allocate output (B, C, H_out, W_out)
        y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=torch.float32)

        # Strides
        stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            pad, pad,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4,
            num_stages=2,
        )

        return y


def run(*args):
    return ModelNew()(*args)
