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
    # Each program computes one output element y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW

            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_offset)
            acc += x_val * w_val

    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version that computes depthwise conv2d (groups=C) using a Triton kernel.
    The rest of the pipeline remains in PyTorch for correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect residual and dwconv_weight as inputs
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects residual and dwconv_weight as inputs.")
        residual = args[0]
        dwconv_weight = args[1]
        pad_h = pad_w = 3

        # Ensure CUDA and contiguous
        device = residual.device
        if device.type != "cuda":
            residual = residual.to("cuda")
            dwconv_weight = dwconv_weight.to("cuda")
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        H_out = H + 2 * pad_h
        W_out = W + 2 * pad_w

        # Output tensor
        y = torch.empty((B, C, H_out, W_out), device=device, dtype=residual.dtype)

        # Extract strides
        stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            pad_h, pad_w,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4,
            num_stages=2,
        )

        # Move back if original was not on CUDA
        if device.type != "cuda":
            return y.to(residual.device)
        return y


def run(*args):
    return ModelNew()(*args)
