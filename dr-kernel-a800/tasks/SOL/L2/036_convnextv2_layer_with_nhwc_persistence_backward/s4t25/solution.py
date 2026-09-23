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

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 depthwise conv with padding=pad_h, pad_w
    for kh in range(0, 7):
        for kw in range(0, 7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Compute input offset with explicit strides
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Load input; if out-of-bounds, use 0.0
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            # Load weight (per-channel scalar)
            w_offset = c * stride_wC + 0 * stride_wKH + kh * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            acc += x_val * w_val

    # Store result to y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        """
        Triton-optimized forward: compute depthwise conv2d (groups=C) using Triton,
        return x_dwconv = conv(residual, dwconv_weight, padding=3).
        """
        assert residual.dim() == 4, "residual must be (B, C, H, W)"
        assert dwconv_weight.dim() == 4, "dwconv_weight must be (C, 1, 7, 7)"
        B, C, H, W = residual.shape
        pad_h = pad_w = 3
        H_out = H + 2 * pad_h
        W_out = W + 2 * pad_w

        # Ensure tensors are on CUDA and contiguous
        device = residual.device
        if device.type != "cuda":
            residual_cuda = residual.contiguous().to("cuda")
            dwconv_weight_cuda = dwconv_weight.contiguous().to("cuda")
            y = torch.empty((B, C, H_out, W_out), device="cuda", dtype=residual.dtype)
        else:
            residual_cuda = residual.contiguous()
            dwconv_weight_cuda = dwconv_weight.contiguous()
            y = torch.empty((B, C, H_out, W_out), device=device, dtype=residual.dtype)

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual_cuda, dwconv_weight_cuda, y,
            B, C, H, W, H_out, W_out,
            pad_h, pad_w,
            residual_cuda.stride(0), residual_cuda.stride(1), residual_cuda.stride(2), residual_cuda.stride(3),
            dwconv_weight_cuda.stride(0), dwconv_weight_cuda.stride(1), dwconv_weight_cuda.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=1, num_stages=1,
        )

        # If input was on CPU, move the result back
        if device.type != "cuda":
            y = y.to(residual.device)

        return y


def run(*args):
    return ModelNew()(*args)
