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

    # Accumulator (float32)
    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 kernel taps
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # Input bounds check
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute input offset
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load weight for channel c and tap (kh, kw)
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            acc += x_val * w_val

    # Store result
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized forward:
    - Uses Triton to compute depthwise conv2d (groups=C) with padding=3.
    - Keeps the rest in torch for simplicity; here we only compute the conv output.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity
        if not residual.is_cuda:
            residual = residual.cuda()
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.cuda()
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        pad = 3
        H_out = H + 2 * pad
        W_out = W + 2 * pad

        # Output tensor
        y = torch.empty((B, C, H_out, W_out), dtype=residual.dtype, device=residual.device)

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
