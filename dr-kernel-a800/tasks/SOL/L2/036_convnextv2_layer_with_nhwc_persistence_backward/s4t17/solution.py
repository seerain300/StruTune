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
    # Grid: one program per output element (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over 7x7 kernel taps
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # Check input bounds
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Compute input offset using strides
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Masked load; if out of bounds, load 0
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            # Load weight for channel c (dwconv_weight shape: [C, 1, 7, 7])
            w_offset = c * stride_wC + 0 * stride_wKH + kh * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            acc += x_val * w_val

    # Store output
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc.to(tl.float32))


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version: compute depthwise conv2d (groups=C) using Triton,
    and keep the rest of the pipeline in PyTorch for robustness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        """
        residual: (B, C, H, W), float32, CUDA
        dwconv_weight: (C, 1, 7, 7), float32, CUDA
        returns x_dwconv: (B, C, H+6, W+6)
        """
        # Ensure CUDA and contiguous
        if not residual.is_cuda:
            residual = residual.to(device="cuda")
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.to(device="cuda")
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        B, C, H, W = residual.shape
        H_out = H + 2 * 3  # padding=3 on both sides
        W_out = W + 2 * 3
        # Output tensor
        y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

        # Strides
        stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            3, 3,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4,  # modest parallelism per program
        )
        return y


def run(*args):
    return ModelNew()(*args)
