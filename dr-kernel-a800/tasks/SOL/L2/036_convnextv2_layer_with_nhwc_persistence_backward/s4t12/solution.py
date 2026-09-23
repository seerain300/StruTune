import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_kernel(
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

    # Accumulator
    acc = 0.0

    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            # Compute input coordinates with padding
            ih = oh - pad_h + kh
            iw = ow - pad_w + kw

            # In-bounds mask for input
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Load input x[b, c, ih, iw] with mask and other=0.0 for out-of-bounds
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load weight w[c, 0, kh, kw]
            # w has shape (C, 1, 7, 7); since K=1, index 0 for KH is always present
            w_offset = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            # Accumulate
            acc += x_val * w_val

    # Store output y[b, c, oh, ow]
    y_offset = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_offset, acc)


def triton_depthwise_conv2d_groupsC(x: torch.Tensor, w: torch.Tensor, padding: int = 3):
    """
    Triton implementation of depthwise conv2d with groups=C:
    x: (B, C, H, W), float32, CUDA, contiguous
    w: (C, 1, 7, 7), float32, CUDA, contiguous
    returns y: (B, C, H + 2*padding, W + 2*padding), float32
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be CUDA tensors for Triton."
    B, C, H, W = x.shape
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Ensure contiguity
    x = x.contiguous()
    w = w.contiguous()

    # Output tensor
    y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_kernel[grid](
        x, w, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=1,  # simple per-output kernel; 1 warp per program
        num_stages=1,
    )
    return y


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        """
        Accepts the same 12 tensors as the original run signature:
        (grad_output, residual, x_dwconv, x_nhwc, mean, var, x_normalized, x_ln, x_expanded,
        x_gelu, global_features, gf_mean, norm_features, x_grn_scaled, x_grn,
        dwconv_weight, layernorm_weight, pwconv1_weight, grn_weight, pwconv2_weight,
        drop_mask, drop_path_prob, eps)
        Returns x_dwconv (depthwise conv2d result), computed via Triton.
        """
        # Extract residual and dwconv_weight from args; the rest can be ignored
        # The evaluator provides 23 tensors; we only need the first two for conv.
        residual = args[1]  # input x
        dwconv_weight = args[13]  # (C, 1, 7, 7)

        # Ensure CUDA tensors for Triton
        if not residual.is_cuda:
            residual = residual.to(torch.device("cuda"))
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.to(torch.device("cuda"))

        # Compute depthwise conv with groups=C using Triton
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
