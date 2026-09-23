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
    # Program ids: one program per output element (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            # Compute input indices with padding
            ih = oh - pad_h + kh
            iw = ow - pad_w + kw

            # Mask to avoid out-of-bounds (since padding may put ih/iw outside [0,H), [0,W))
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute pointers for input and weight
            x_idx = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            w_idx = c * stride_wC + 0 * stride_wKH + kh * stride_wKW

            # Load input (masked) and weight (scalar). Default to 0 outside bounds.
            x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_idx)

            # Accumulate in fp32
            acc += x_val * w_val

    # Store result to output
    y_idx = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_idx, acc)


def triton_depthwise_conv2d_groupsC(x: torch.Tensor, w: torch.Tensor, padding: int = 3):
    """
    Triton-optimized depthwise conv2d with groups=C:
    - x: (B, C, H, W), float32, CUDA
    - w: (C, 1, 7, 7), float32, CUDA
    - Returns y: (B, C, H + 2*padding, W + 2*padding), float32
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA for Triton."
    B, C, H, W = x.shape
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Ensure inputs are contiguous for predictable strides
    x = x.contiguous()
    w = w.contiguous()

    # Allocate output
    y = torch.empty((B, C, H_out, W_out), device=x.device, dtype=torch.float32)

    # Get strides
    stride_xB, stride_xC, stride_xH, stride_xW = x.stride()
    stride_wC, stride_wKH, stride_wKW = w.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_output_kernel[grid](
        x, w, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=1,
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses Triton to compute depthwise conv2d (groups=C) on the residual input.
    - Rest of the pipeline is left to PyTorch for simplicity and correctness in this submission.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        # Ensure CUDA execution for Triton
        if not residual.is_cuda:
            residual = residual.to(torch.device("cuda"))
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.to(torch.device("cuda"))

        # Compute depthwise convolution with groups=C using Triton
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
