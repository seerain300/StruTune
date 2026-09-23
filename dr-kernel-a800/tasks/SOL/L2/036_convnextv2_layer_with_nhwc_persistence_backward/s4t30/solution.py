import torch
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per-output depthwise conv with groups=C
@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # program ids: (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # accumulator (fp32)
    acc = tl.zeros((), dtype=tl.float32)

    # loop over 7x7 kernel taps
    for kh in tl.static_range(0, 7):
        for kw in tl.static_range(0, 7):
            # compute input indices with padding
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # bounds check for input
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # load input element (masked)
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            # load weight scalar for channel c at (kh, kw)
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)
            # accumulate
            acc += x_val * w_val

    # store result to y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor, H: int, W: int):
        """
        Triton-optimized forward:
        - If Triton and CUDA are available, compute depthwise conv2d (groups=C) using a Triton kernel.
        - Otherwise, fall back to torch.nn.functional.conv2d to ensure correctness.
        Returns x_dwconv with shape (B, C, H+6, W+6).
        """
        # Fallback if Triton/CUDA not available
        use_triton = TRITON_AVAILABLE and residual.is_cuda and dwconv_weight.is_cuda

        if not use_triton:
            # Use PyTorch reference implementation
            x_dwconv = F.conv2d(residual, dwconv_weight, bias=None, padding=3, groups=C)
            return x_dwconv

        # Triton path: ensure contiguous inputs/weights
        residual_c = residual.contiguous()
        dwconv_weight_c = dwconv_weight.contiguous()
        B, C_channels, H_in, W_in = residual_c.shape
        # output size for depthwise conv with padding=3
        H_out = H_in + 2 * 3
        W_out = W_in + 2 * 3

        # Allocate output
        x_dwconv = torch.empty((B, C_channels, H_out, W_out), dtype=residual_c.dtype, device=residual_c.device)

        # Strides
        stride_xB, stride_xC, stride_xH, stride_xW = residual_c.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight_c.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = x_dwconv.stride()

        # Launch one program per output element
        grid = (B, C_channels, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual_c, dwconv_weight_c, x_dwconv,
            B, C_channels, H_in, W_in, H_out, W_out,
            3, 3,  # padding
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4, num_stages=2,
        )
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
