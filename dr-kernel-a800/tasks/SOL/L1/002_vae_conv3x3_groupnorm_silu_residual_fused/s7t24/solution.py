import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,                # *float32 input tensor (B, C_in, H, W)
    w_ptr,                # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,                # *float32 output tensor (B, C_out, H, W)
    N, C_in, C_out, H, W,             # sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ic, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC), ceil_div(H*W, BLOCK_HW))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    hw_block_id = tl.program_id(2)

    oc_start = oc_block_id * BLOCK_OC
    hw_start = hw_block_id * BLOCK_HW

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # accumulator for output channels in tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # iterate over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # flatten hw indices for this block
                hw_idx = hw_start + tl.arange(0, BLOCK_HW)
                mask_hw = hw_idx < (H * W)
                h = hw_idx // W
                w = hw_idx % W

                # compute input coordinates with padding=1
                ih = h + kh - 1
                iw = w + kw - 1

                # valid positions are within H and W
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_hw

                # load input values
                x_ptrs = n * x_stride_n + cin * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_vals = tl.load(x_ptr + x_ptrs, mask=valid, other=0.0)

                # load weights for oc tile
                w_ptrs = oc_offsets * w_stride_oc + cin * w_stride_ic + kh * w_stride_kh + kw * w_stride_kw
                w_vals = tl.load(w_ptr + w_ptrs, mask=oc_mask, other=0.0)

                # accumulate
                for j in range(BLOCK_OC):
                    if oc_mask[j]:
                        acc[j] += tl.sum(x_vals * w_vals[j], axis=0)

    # store results for all hw in the block
    for j in range(BLOCK_OC):
        if oc_mask[j]:
            # acc[j] is computed for all hw in the block; we need to write it per (h, w)
            # We can store acc[j] into y[n, oc, h, w] for each hw index in this block.
            for hw in range(BLOCK_HW):
                if (hw_start + hw) < (H * W):
                    h = (hw_start + hw) // W
                    w = (hw_start + hw) % W
                    y_ptrs = n * y_stride_n + j * y_stride_c + h * y_stride_h + w * y_stride_w
                    tl.store(y_ptr + y_ptrs, acc[j])


@triton.jit
def group_norm_affine_kernel(
    x_ptr,                # input [N, C, H, W] with strides
    weight_ptr,           # per-channel scale [C] with strides
    bias_ptr,             # per-channel bias [C] with strides
    y_ptr,                # output [N, C, H, W] with strides
    N, C, H, W,           # sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    weight_stride_c, bias_stride_c,
    num_groups: tl.constexpr,        # number of groups (32)
    eps,                          # epsilon for numerical stability
    BLOCK_HW: tl.constexpr,        # spatial tile (e.g., 4096)
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # pass 1: compute sum and sumsq over the group
    sum_g = 0.0
    sumsq_g = 0.0

    c = group_start
    while c < group_start + channels_per_group:
        hw = 0
        while hw < H * W:
            hw_idx = hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < (H * W)
            h = hw_idx // W
            w = hw_idx % W
            x_ptrs = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
            x_vals = tl.load(x_ptr + x_ptrs, mask=mask_hw, other=0.0)
            sum_g += tl.sum(x_vals, axis=0)
            sumsq_g += tl.sum(x_vals * x_vals, axis=0)
            hw += BLOCK_HW
        c += 1

    mean = sum_g / (channels_per_group * H * W)
    var = sumsq_g / (channels_per_group * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize and apply affine
    c = group_start
    while c < group_start + channels_per_group:
        weight_val = tl.load(weight_ptr + c * weight_stride_c)
        bias_val = tl.load(bias_ptr + c * bias_stride_c)
        hw = 0
        while hw < H * W:
            hw_idx = hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < (H * W)
            h = hw_idx // W
            w = hw_idx % W
            x_ptrs = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
            x_vals = tl.load(x_ptr + x_ptrs, mask=mask_hw, other=0.0)
            y_vals = (x_vals - mean) * inv_std
            y_vals = y_vals * weight_val + bias_val
            y_ptrs = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            tl.store(y_ptr + y_ptrs, y_vals, mask=mask_hw)
            hw += BLOCK_HW
        c += 1


@triton.jit
def silu_kernel_with_total(N, C, H, W, total, y_ptr, x_ptr):
    # One program per n; loop over all elements for this n
    n = tl.program_id(0)
    i = 0
    while i < total:
        # compute h, w from i
        W_local = W  # H, W should be passed correctly; we use total only
        # Since total = N * C * H * W is not available here, we rely on grid launch across N
        # Each program will iterate over its own elements using global i but with correct total.
        # We re-define simple elementwise per program using total from launch:
        x_val = tl.load(x_ptr + i)
        # sigmoid
        sig = 1.0 / (1.0 + tl.exp(-x_val))
        y_val = x_val * sig
        tl.store(y_ptr + i, y_val)
        i += 1


@triton.jit
def add_residual_kernel(N, C, H, W, total, y_ptr, x_ptr, residual_ptr):
    n = tl.program_id(0)
    i = 0
    while i < total:
        val = tl.load(y_ptr + i)
        res = tl.load(residual_ptr + i)
        out = val + res
        tl.store(y_ptr + i, out)
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure inputs are contiguous and float32 for Triton
        device = x.device
        x = x.contiguous().to(torch.float32)

        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        N, C, H, W = x.shape

        # 1) Conv1 (Triton)
        y1 = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        conv3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 32), triton.cdiv(H * W, 4096))](
            x, conv1_weight, y1,
            N, C, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_HW=4096, num_warps=4, num_stages=2
        )

        # 2) GroupNorm1 (Triton)
        y1_norm = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        group_norm_affine_kernel[(N, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1_norm,
            N, C, H, W,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            norm1_weight.stride(0), norm1_bias.stride(0),
            num_groups=self.num_groups, eps=self.eps, num_warps=4, num_stages=2
        )

        # 3) SiLU1 (Triton)
        y1_silu = torch.empty_like(y1_norm)
        total_elems = N * C * H * W
        # Launch one program per (N); this simple kernel loops over all elements of its n slice.
        silu_kernel_with_total[(N,)](
            N, C, H, W, total_elems, y1_silu, y1_norm, num_warps=4, num_stages=2
        )

        # Save residual x for addition
        residual = x

        # 4) Conv2 (Triton)
        y2 = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        conv3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 32), triton.cdiv(H * W, 4096))](
            y1_silu, conv2_weight, y2,
            N, C, C, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=32, BLOCK_HW=4096, num_warps=4, num_stages=2
        )

        # 5) GroupNorm2 (Triton)
        y2_norm = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        group_norm_affine_kernel[(N, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2_norm,
            N, C, H, W,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            norm2_weight.stride(0), norm2_bias.stride(0),
            num_groups=self.num_groups, eps=self.eps, num_warps=4, num_stages=2
        )

        # 6) SiLU2 (Triton)
        y2_silu = torch.empty_like(y2_norm)
        silu_kernel_with_total[(N,)](
            N, C, H, W, total_elems, y2_silu, y2_norm, num_warps=4, num_stages=2
        )

        # 7) Add residual (Triton)
        out = torch.empty_like(y2_silu)
        add_residual_kernel[(N,)](
            N, C, H, W, total_elems, out, y2_silu, residual, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
