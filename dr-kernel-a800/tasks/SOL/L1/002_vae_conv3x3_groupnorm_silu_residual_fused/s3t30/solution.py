import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over H*W). Each program handles one (b, oc) and one spatial tile.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_SP: tl.constexpr,  # e.g., 256
    BLOCK_IC: tl.constexpr,  # e.g., 32
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    tile_id = tl.program_id(2)

    # spatial tile
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)
    h = sp // W
    w = sp % W

    # accumulator for this spatial tile
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_IC):
        # For each input channel in this block, accumulate over 3x3 neighborhood
        for ic in range(0, BLOCK_IC):
            ic_idx = ic_base + ic
            ic_valid = ic_idx < C_in

            # kh = -1: ih = h - 1, iw = w - 1
            ih = h + (-1)
            iw = w + (-1)
            x_ptrs_kh_neg = x_ptr + b * x_stride_b + ic_idx * x_stride_c + ih * x_stride_h + iw * x_stride_w
            x_vals_kh_neg = tl.load(x_ptrs_kh_neg, mask=sp_mask & ic_valid, other=0.0)

            w_ptrs_kh_neg = w_ptr + ic_idx * w_stride_cin + oc * w_stride_cout + (-1) * w_stride_kh + 0 * w_stride_kw
            w_vals_kh_neg = tl.load(w_ptrs_kh_neg, mask=ic_valid, other=0.0)

            acc += x_vals_kh_neg * w_vals_kh_neg

            # kh = 0: ih = h, iw = w
            ih0 = h
            iw0 = w
            x_ptrs_kh0 = x_ptr + b * x_stride_b + ic_idx * x_stride_c + ih0 * x_stride_h + iw0 * x_stride_w
            x_vals_kh0 = tl.load(x_ptrs_kh0, mask=sp_mask & ic_valid, other=0.0)

            w_ptrs_kh0 = w_ptr + ic_idx * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + 0 * w_stride_kw
            w_vals_kh0 = tl.load(w_ptrs_kh0, mask=ic_valid, other=0.0)

            acc += x_vals_kh0 * w_vals_kh0

            # kh = 1: ih = h + 1, iw = w
            ih1 = h + 1
            iw1 = w
            x_ptrs_kh1 = x_ptr + b * x_stride_b + ic_idx * x_stride_c + ih1 * x_stride_h + iw1 * x_stride_w
            x_vals_kh1 = tl.load(x_ptrs_kh1, mask=sp_mask & ic_valid, other=0.0)

            w_ptrs_kh1 = w_ptr + ic_idx * w_stride_cin + oc * w_stride_cout + 1 * w_stride_kh + 0 * w_stride_kw
            w_vals_kh1 = tl.load(w_ptrs_kh1, mask=ic_valid, other=0.0)

            acc += x_vals_kh1 * w_vals_kh1

            # kw = -1: ih = h, iw = w - 1
            iw_kw_neg = w + (-1)
            x_ptrs_kw_neg = x_ptr + b * x_stride_b + ic_idx * x_stride_c + h * x_stride_h + iw_kw_neg * x_stride_w
            x_vals_kw_neg = tl.load(x_ptrs_kw_neg, mask=sp_mask & ic_valid, other=0.0)

            w_ptrs_kw_neg = w_ptr + ic_idx * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + (-1) * w_stride_kw
            w_vals_kw_neg = tl.load(w_ptrs_kw_neg, mask=ic_valid, other=0.0)

            acc += x_vals_kw_neg * w_vals_kw_neg

            # kw = 1: ih = h, iw = w + 1
            iw_kw_pos = w + 1
            x_ptrs_kw_pos = x_ptr + b * x_stride_b + ic_idx * x_stride_c + h * x_stride_h + iw_kw_pos * x_stride_w
            x_vals_kw_pos = tl.load(x_ptrs_kw_pos, mask=sp_mask & ic_valid, other=0.0)

            w_ptrs_kw_pos = w_ptr + ic_idx * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + 1 * w_stride_kw
            w_vals_kw_pos = tl.load(w_ptrs_kw_pos, mask=ic_valid, other=0.0)

            acc += x_vals_kw_pos * w_vals_kw_pos

    # store results
    y_ptrs = y_ptr + b * y_stride_b + oc * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


def _run_conv3x3_triton(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), out: (B, C_out, H, W)
    B, C_in, H, W = x.shape
    C_out = w.shape[1]

    # Allocate output
    out = torch.empty((B, C_out, H, W), dtype=torch.float32, device=x.device)

    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw = w.stride()
    y_stride_b, y_stride_c, y_stride_h, y_stride_w = out.stride()

    # Grid: (B, C_out, tiles over H*W)
    BLOCK_SP = 256
    tiles_sp = (H * W + BLOCK_SP - 1) // BLOCK_SP
    grid = (B, C_out, tiles_sp)
    conv3x3_triton[grid](
        x, w, out,
        B, C_in, H, W, C_out,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w,
        BLOCK_SP=BLOCK_SP,
        BLOCK_IC=32,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure CUDA and contiguous
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "All tensors must be CUDA tensors"

        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups"

        # Conv1
        out1 = _run_conv3x3_triton(x, conv1_weight)

        # GroupNorm + SiLU after conv1
        out1 = F.group_norm(out1, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)
        out1 = F.silu(out1)

        # Conv2
        out2 = _run_conv3x3_triton(out1, conv2_weight)

        # GroupNorm + SiLU after conv2
        out2 = F.group_norm(out2, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)
        out2 = F.silu(out2)

        # Residual add: out = out2 + x
        N = B * C * H * W
        out = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        BLOCK_RES = 1024
        grid_res = ((N + BLOCK_RES - 1) // BLOCK_RES,)
        add_residual_kernel[grid_res](out, out2, x, N, BLOCK_RES, num_warps=4, num_stages=2)
        return out


# Original run function for reference
@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    num_groups = 32
    residual = x

    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out = F.silu(out)

    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out = F.silu(out)

    out = out + residual
    return out


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
