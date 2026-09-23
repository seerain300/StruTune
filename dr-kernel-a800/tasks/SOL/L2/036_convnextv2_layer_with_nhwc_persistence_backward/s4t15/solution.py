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
    # Program ids correspond to (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 depthwise convolution with padding
    for kh in range(0, 7):
        for kw in range(0, 7):
            # Compute corresponding input coordinates with padding
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # In-bounds check for input
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Load input value with mask; out-of-bounds get 0
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load per-channel weight scalar w[c, 0, kh, kw]
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            acc += x_val * w_val

    # Store to output y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Computes depthwise conv2d with groups=C using a Triton kernel.
    - The rest of the pipeline is done in PyTorch to preserve exact semantics and correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        """
        residual: (B, C, H, W)
        dwconv_weight: (C, 1, 7, 7)
        returns x_dwconv: (B, C, H+6, W+6)
        """
        # Ensure CUDA tensors for Triton
        if not residual.is_cuda:
            residual = residual.cuda()
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.cuda()

        B, C, H, W = residual.shape
        H_out = H + 2 * 3  # padding=3
        W_out = W + 2 * 3  # padding=3

        # Allocate output
        y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

        # Make tensors contiguous for simple stride handling
        x = residual.contiguous()
        w = dwconv_weight.contiguous()
        y = y.contiguous()

        # Strides
        stride_xB, stride_xC, stride_xH, stride_xW = x.stride()
        stride_wC, stride_wKH, stride_wKW = w.stride()  # for (C,1,7,7): last dim is 7
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            x, w, y,
            B, C, H, W, H_out, W_out,
            3, 3,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=2,
            num_stages=2,
        )

        # If original input was on CPU, move result back
        if residual.device.type != 'cuda':
            y = y.cpu()

        return y


def run(*args):
    return ModelNew()(*args)
