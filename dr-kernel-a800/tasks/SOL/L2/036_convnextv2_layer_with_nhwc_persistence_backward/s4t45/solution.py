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
    # One program per output element (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel taps
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # Mask for valid input indices
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Compute input pointer offset (x has strides (B,C,H,W))
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Load input (masked); other=0 ensures out-of-bounds contribute 0
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            # Load weight: weight is per-channel, shape (C, 1, 7, 7)
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            acc += x_val * w_val

    # Store result to output (y has strides (B,C,H_out,W_out))
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized model: computes depthwise conv2d with groups=C using a Triton kernel,
    and returns the conv output tensor. The rest of the pipeline remains in PyTorch for robustness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        # Ensure tensors are on CUDA for Triton
        if residual.device.type != "cuda":
            residual = residual.to("cuda")
        if dwconv_weight.device.type != "cuda":
            dwconv_weight = dwconv_weight.to("cuda")

        # Shapes
        B, C, H, W = residual.shape
        pad_h = pad_w = 3
        H_out = H + 2 * pad_h
        W_out = W + 2 * pad_w

        # Make inputs/weights contiguous to simplify stride handling
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        # Allocate output tensor (float32 for numerical stability)
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
            pad_h, pad_w,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=2,
            num_stages=2,
        )

        return y


def run(*args):
    return ModelNew()(*args)
