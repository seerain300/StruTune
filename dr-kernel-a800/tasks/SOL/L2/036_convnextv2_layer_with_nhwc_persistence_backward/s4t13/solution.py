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

    # 7x7 depthwise kernel
    for kh in range(7):
        for kw in range(7):
            ih = oh - pad_h + kh
            iw = ow - pad_w + kw

            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Address input with explicit strides
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # Address per-channel weight: w[c, 0, kh, kw]
            w_off = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_off)

            acc += x_val * w_val

    # Store output y[b, c, oh, ow]
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


def triton_depthwise_conv2d_groupsC(x: torch.Tensor, w: torch.Tensor, padding: int = 3):
    """
    Triton implementation of depthwise conv2d with groups=C:
    x: (B, C, H, W), w: (C, 1, 7, 7), no bias.
    Returns y: (B, C, H + 2*padding, W + 2*padding).
    """
    assert x.is_cuda and w.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dim() == 4 and w.dim() == 4, "x must be (B,C,H,W), w must be (C,1,7,7)"
    B, C, H, W = x.shape
    kH, kW = 7, 7
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Ensure contiguous for simple stride-based addressing
    x_c = x.contiguous()
    w_c = w.contiguous()

    # Output allocation
    y = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_output_kernel[grid](
        x_c, w_c, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=4,  # modest parallelism per program
        num_stages=2, # modest pipelining
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version focusing on depthwise conv2d (groups=C).
    Uses Triton to compute x_dwconv and returns it.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        # Move to CUDA if needed (Triton requires CUDA)
        if not residual.is_cuda:
            residual = residual.to(device=torch.device("cuda"))
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.to(device=torch.device("cuda"))

        # Compute depthwise conv using Triton
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
