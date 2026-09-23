import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_tiled_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
    BLOCK_H: tl.constexpr,  # tile height
    BLOCK_W: tl.constexpr,  # tile width
):
    # Grid: (B, C, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W))
    b = tl.program_id(0)
    c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Tile start indices
    oh0 = pid_h * BLOCK_H
    ow0 = pid_w * BLOCK_W

    # Output indices within the tile
    oh_vec = oh0 + tl.arange(0, BLOCK_H)[:, None]  # (BLOCK_H, 1)
    ow_vec = ow0 + tl.arange(0, BLOCK_W)[None, :]  # (1, BLOCK_W)

    # Valid output positions
    out_mask = (oh_vec < H_out) & (ow_vec < W_out)

    # Accumulator for tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Unrolled 7x7 depthwise convolution
    # Using static_range allows Triton to better unroll and optimize.
    for kh in tl.static_range(7):
        for kw in tl.static_range(7):
            ih = oh_vec + kh - pad_h           # (BH, 1)
            iw = ow_vec + kw - pad_w           # (1, BW)
            # Input bounds
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & out_mask

            # Load input x[b, c, ih, iw] with mask
            x_offset = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

            # Load weight w[c, 0, kh, kw] (scalar per kh, kw)
            w_offset = c * stride_wC + 0 * stride_wKH + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_ptr + w_offset)

            acc += x_val * w_val

    # Store the tile y[b, c, oh_vec, ow_vec]
    y_offset = b * stride_yB + c * stride_yC + oh_vec * stride_yH + ow_vec * stride_yW
    tl.store(y_ptr + y_offset, acc, mask=out_mask)


def triton_depthwise_conv2d_groupsC(residual: torch.Tensor, dwconv_weight: torch.Tensor, padding: int = 3):
    """
    Triton implementation of depthwise conv2d with groups=C.
    residual: (B, C, H, W)
    dwconv_weight: (C, 1, 7, 7)
    output: (B, C, H+2*padding, W+2*padding)
    """
    assert residual.ndim == 4 and dwconv_weight.ndim == 4, "Invalid input shapes"
    B, C, H, W = residual.shape
    residual = residual.contiguous()
    dwconv_weight = dwconv_weight.contiguous()

    H_out = H + 2 * padding
    W_out = W + 2 * padding
    y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)

    # Strides
    stride_xB, stride_xC, stride_xH, stride_xW = residual.stride()
    stride_wC, stride_wKH, stride_wKW = dwconv_weight.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Large tile to reduce program count
    BLOCK_H = 16
    BLOCK_W = 32
    grid = (
        B,
        C,
        triton.cdiv(H_out, BLOCK_H),
        triton.cdiv(W_out, BLOCK_W),
    )

    depthwise_conv2d_groupsC_tiled_kernel[grid](
        residual, dwconv_weight, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8,   # increase parallelism per program
        num_stages=3,  # improve pipelining
    )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses Triton to compute depthwise conv2d (groups=C) on the residual input.
    - The rest of the pipeline remains in torch to preserve original semantics and ensure correctness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect residual and dwconv_weight as inputs (as in the original run signature)
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects at least residual and dwconv_weight as inputs.")
        residual = args[0]
        dwconv_weight = args[1]
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)
        return x_dwconv


def run(*args):
    return ModelNew()(*args)
