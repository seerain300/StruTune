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
    # Program ids for (b, c, oh, ow)
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel taps
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w

            # Check input bounds
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            # Compute input offset
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Load input with mask; out-of-bounds treated as 0
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # Load weight and multiply
            w_off = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_off)
            acc += x_val * w_val

    # Store output
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor):
        """
        Compute depthwise conv2d (groups=C) using Triton and return the result.
        residual: (B, C, H, W), dwconv_weight: (C, 1, 7, 7)
        Returns: y: (B, C, H+6, W+6)
        """
        assert residual.dim() == 4 and dwconv_weight.dim() == 4
        B, C, H, W = residual.shape
        kH, kW = 7, 7
        pad_h = pad_w = 3
        H_out = H + 2 * pad_h
        W_out = W + 2 * pad_w

        # Ensure tensors are on CUDA and contiguous for Triton
        device = residual.device
        residual_c = residual.contiguous()
        dwconv_weight_c = dwconv_weight.contiguous()

        # Output tensor (float32 accumulation is done in kernel; we store float32)
        y = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual_c, dwconv_weight_c, y,
            B, C, H, W, H_out, W_out,
            pad_h, pad_w,
            residual_c.stride(0), residual_c.stride(1), residual_c.stride(2), residual_c.stride(3),
            dwconv_weight_c.stride(0), dwconv_weight_c.stride(1), dwconv_weight_c.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=1, num_stages=1,
        )

        return y


def run(*args):
    return ModelNew()(*args)
