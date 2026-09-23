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

    # Accumulator (fp32)
    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 depthwise kernel
    for kh in range(7):
        for kw in range(7):
            # Compute input coordinates with padding
            in_h = oh + kh - pad_h
            in_w = ow + kw - pad_w

            # Bounds check for input
            in_bounds = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

            # Load input: x[b, c, in_h, in_w]
            x_offset = b * stride_xB + c * stride_xC + in_h * stride_xH + in_w * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load weight: w[c, 0, kh, kw]
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            # Accumulate
            acc += x_val * w_val

    # Store output: y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute depthwise conv2d with groups=C using Triton:
        y = conv2d(residual, dwconv_weight, padding=3, groups=C), no bias.
        Returns y with shape (B, C, H_out, W_out), where H_out = H + 2*pad, W_out = W + 2*pad.
        """
        # Ensure CUDA for Triton
        device = residual.device
        resid = residual.detach()
        if device.type != "cuda":
            resid = resid.to("cuda")
        w = dwconv_weight.detach()
        if w.device.type != "cuda":
            w = w.to("cuda")

        # Contiguous for simple strides
        resid = resid.contiguous()
        w = w.contiguous()

        B, C, H, W = resid.shape
        pad = 3
        H_out = H + 2 * pad
        W_out = W + 2 * pad

        # Output tensor (float32)
        y = torch.empty((B, C, H_out, W_out), dtype=torch.float32, device=resid.device)

        # Launch: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            resid, w, y,
            B, C, H, W, H_out, W_out,
            pad, pad,
            resid.stride(0), resid.stride(1), resid.stride(2), resid.stride(3),
            w.stride(0), w.stride(1), w.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=1,
            num_stages=1,
        )
        return y


def run(*args):
    return ModelNew()(*args)
