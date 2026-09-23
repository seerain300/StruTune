import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # One program per output element y[b, c, oh, ow]
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel; padding=3 is handled by input indexing
    for kh in range(0, 7):
        for kw in range(0, 7):
            ih = oh + kh - 3
            iw = ow + kw - 3

            # Check bounds for input
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute input offset with strides
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # Load weight w[c, 0, kh, kw]
            w_off = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_off)

            acc += x_val * w_val

    # Store result to y[b, c, oh, ow]
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


def triton_depthwise_conv2d_groupsC(x: torch.Tensor, w: torch.Tensor, padding: int = 3) -> torch.Tensor:
    """
    Triton implementation of depthwise conv2d with groups=C:
    Input x: (B, C, H, W), weight w: (C, 1, 7, 7)
    Output y: (B, C, H_out, W_out), H_out = H + 2*padding, W_out = W + 2*padding
    """
    assert x.dim() == 4 and w.dim() == 4, "x must be (B,C,H,W), w must be (C,1,7,7)"
    B, C, H, W = x.shape
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Ensure contiguous and on CUDA
    x_c = x.contiguous()
    w_c = w.contiguous()
    device = x.device
    if device.type != "cuda":
        device = torch.device("cuda")
    y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=device)

    # Get strides
    stride_xB, stride_xC, stride_xH, stride_xW = x_c.stride()
    stride_wC, stride_wKH, stride_wKW = w_c.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_output_kernel[grid](
        x_c, w_c, y,
        B, C, H, W, H_out, W_out,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=4,  # small parallelism is fine for per-output kernel
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Computes depthwise conv2d (groups=C, padding=3) using a Triton kernel.
    - Returns the convolution output tensor.
    """
    def forward(self, *args):
        # Expect residual and dwconv_weight as inputs
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects at least residual and dwconv_weight as inputs.")
        residual = args[0]
        dwconv_weight = args[1]
        # Ensure CUDA tensors
        if residual.device.type != "cuda":
            residual = residual.to("cuda")
        if dwconv_weight.device.type != "cuda":
            dwconv_weight = dwconv_weight.to("cuda")
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
