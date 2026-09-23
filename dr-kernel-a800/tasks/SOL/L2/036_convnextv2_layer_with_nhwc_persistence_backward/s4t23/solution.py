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
    # Grid: (B, C, H_out, W_out) — one program per output element
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # in-bounds check
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Load input with mask; cast to float32
            x_addr = x_ptr + b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_addr, mask=in_bounds, other=0.0)
            x_val = x_val.to(tl.float32)

            # Load weight; per-channel
            w_addr = w_ptr + c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_addr)
            w_val = w_val.to(tl.float32)

            acc += x_val * w_val

    # Store result to y[b, c, oh, ow]
    y_addr = y_ptr + b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_addr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor, H: int, W: int):
        """
        residual: (B, C, H, W), float32 or float16; can be on CPU or CUDA.
        dwconv_weight: (C, 1, 7, 7), same dtype as residual, can be on CPU or CUDA.
        Returns y: (B, C, H_out, W_out), where H_out = H + 2*pad, W_out = W + 2*pad, pad=3.
        """
        pad = 3
        H_out = H + 2 * pad
        W_out = W + 2 * pad

        # Ensure tensors are on CUDA for Triton
        if not residual.is_cuda:
            residual = residual.cuda(non_blocking=True)
        if not dwconv_weight.is_cuda:
            dwconv_weight = dwconv_weight.cuda(non_blocking=True)

        # Make inputs contiguous
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        # Allocate output
        B, C, H_in, W_in = residual.shape
        y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

        # Get strides (in elements)
        stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
        stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
        stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

        # Launch Triton kernel: one program per output element
        grid = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid](
            residual, dwconv_weight, y,
            B, C, H_in, W_in, H_out, W_out,
            pad, pad,
            stride_xB, stride_xC, stride_xH, stride_xW,
            stride_wC, stride_wKH, stride_wKW,
            stride_yB, stride_yC, stride_yH, stride_yW,
            num_warps=4,  # modest parallelism per program
            num_stages=1, # simple kernel; pipelining not critical
        )

        # If original residual was on CPU, move output back to CPU
        if not residual.is_cuda:
            y = y.cpu()

        return y


def run(*args):
    return ModelNew()(*args)
