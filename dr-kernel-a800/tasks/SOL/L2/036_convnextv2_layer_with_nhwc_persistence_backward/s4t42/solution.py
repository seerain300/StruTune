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
    # One program per output element: y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in float32
    acc = 0.0

    # Loop over 7x7 kernel
    for kh in range(0, 7):
        for kw in range(0, 7):
            # Compute input coordinates with padding
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # Check bounds; if out-of-bounds, treat as zero
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute input offset and load (masked)
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load weight w[c, 0, kh, kw]
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            # Accumulate
            acc += x_val * w_val

    # Store the accumulated result to y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version that uses a Triton kernel to compute
    depthwise conv2d with groups=C. The rest of the pipeline is kept in torch
    to ensure correctness and avoid runtime errors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous
        if device is None:
            device = residual.device
        if residual.device.type != 'cuda':
            residual = residual.to('cuda')
        if dwconv_weight.device.type != 'cuda':
            dwconv_weight = dwconv_weight.to('cuda')

        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        pad = 3
        H_out = H + 2 * pad
        W_out = W + 2 * pad

        # Allocate output
        y = torch.empty((B, C, H_out, W_out), device='cuda', dtype=torch.float32)

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

        # Return conv output
        return y


def run(*args):
    return ModelNew()(*args)
