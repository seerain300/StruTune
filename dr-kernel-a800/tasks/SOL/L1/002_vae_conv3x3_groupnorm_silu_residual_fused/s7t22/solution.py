import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,                # *float32 input tensor (B, C_in, H, W)
    w_ptr,                # *float32 weights tensor (C_in, C_out, 3, 3)
    y_ptr,                # *float32 output tensor (B, C_out, H, W)
    N, C_in, C_out, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,        # tile of output channels per program
    BLOCK_HW: tl.constexpr,        # spatial tile size
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC), ceil_div(H*W, BLOCK_HW))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    hw_block_id = tl.program_id(2)

    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Spatial tile for storing
    hw_start = hw_block_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < (H * W)
    h = hw_offsets // W
    w = hw_offsets % W

    # Accumulator for output channels in tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Compute input coordinates with padding=1
                ih = h + kh - 1
                iw = w + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & hw_mask

                # Linear index into x: (((n * C_in + cin) * H + ih) * W + iw)
                in_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_vals = tl.load(x_ptr + in_index, mask=in_bounds, other=0.0)

                # Load weights for all oc in tile: (((cin * C_out + oc) * 9) + (kh * 3 + kw))
                for j in range(BLOCK_OC):
                    if oc_mask[j]:
                        w_index = (((cin * C_out + oc_offsets[j]) * 9) + (kh * 3 + kw))
                        w_val = tl.load(w_ptr + w_index)
                        acc[j] += x_vals * w_val

    # Store results to y[n, oc, h, w] for all oc in tile
    # y linear index: n*y_stride_n + oc*y_stride_c + h*y_stride_h + w*y_stride_w
    for j in range(BLOCK_OC):
        if oc_mask[j]:
            y_ptrs = n * y_stride_n + oc_offsets[j] * y_stride_c + h * y_stride_h + w * y_stride_w
            tl.store(y_ptr + y_ptrs, acc[j], mask=hw_mask)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,                # *float32 input tensor [N, C, H, W]
    weight_ptr,           # *float32 per-channel scale [C]
    bias_ptr,             # *float32 per-channel bias [C]
    y_ptr,                # *float32 output tensor [N, C, H, W]
    N, C, H, W,           # sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    weight_stride_c, bias_stride_c,
    num_groups: tl.constexpr,        # number of groups (32)
    eps,                          # epsilon for numerical stability
    BLOCK_HW: tl.constexpr,        # spatial tile size
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Pass 1: compute sum and sumsq over the group
    sum_g = 0.0
    sumsq_g = 0.0

    cin = 0
    while cin < channels_per_group:
        c = group_start + cin
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
        cin += 1

    mean = sum_g / (channels_per_group * H * W)
    var = sumsq_g / (channels_per_group * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
    cin = 0
    while cin < channels_per_group:
        c = group_start + cin
        hw = 0
        while hw < H * W:
            hw_idx = hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < (H * W)
            h = hw_idx // W
            w = hw_idx % W

            x_ptrs = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
            x_vals = tl.load(x_ptr + x_ptrs, mask=mask_hw, other=0.0)

            scale = tl.load(weight_ptr + c * weight_stride_c)
            bias = tl.load(bias_ptr + c * bias_stride_c)

            y_vals = (x_vals - mean) * inv_std
            y_vals = y_vals * scale + bias

            y_ptrs = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            tl.store(y_ptr + y_ptrs, y_vals, mask=mask_hw)

            hw += BLOCK_HW
        cin += 1


@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_HW: tl.constexpr,
):
    # Simple elementwise kernel over N, C, H, W
    # We launch over a 1D grid and iterate over H*W in tiles.
    linear = tl.program_id(0)
    total = N * C * H * W
    start = linear * BLOCK_HW
    offsets = start + tl.arange(0, BLOCK_HW)
    mask = offsets < total

    # Compute n, c, h, w from linear offsets
    rem1 = offsets % W
    hw = offsets % (H * W)
    n = offsets // (C * H * W)
    c = (offsets % (C * H * W)) // (H * W)
    h = hw // W
    w = hw % W

    x_ptrs = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    y_ptrs = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w

    x_vals = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_vals))
    y_vals = x_vals * sig

    tl.store(y_ptr + y_ptrs, y_vals, mask=mask)


@triton.jit
def add_residual_kernel(
    y_ptr, x_ptr, out_ptr,
    N, C, H, W,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_HW: tl.constexpr,
):
    # Elementwise add: out = y + x
    linear = tl.program_id(0)
    total = N * C * H * W
    start = linear * BLOCK_HW
    offsets = start + tl.arange(0, BLOCK_HW)
    mask = offsets < total

    rem1 = offsets % W
    hw = offsets % (H * W)
    n = offsets // (C * H * W)
    c = (offsets % (C * H * W)) // (H * W)
    h = hw // W
    w = hw % W

    y_ptrs = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    x_ptrs = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    out_ptrs = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

    y_vals = tl.load(y_ptr + y_ptrs, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
    out_vals = y_vals + x_vals

    tl.store(out_ptr + out_ptrs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
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
        y1_norm = torch.empty_like(y1)
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
        silu_kernel[(triton.cdiv(total_elems, 4096),)](
            y1_norm, y1_silu,
            N, C, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            num_warps=4, num_stages=2
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
        y2_norm = torch.empty_like(y2)
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
        silu_kernel[(triton.cdiv(total_elems, 4096),)](
            y2_norm, y2_silu,
            N, C, H, W,
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Add residual (Triton)
        y_out = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        add_residual_kernel[(triton.cdiv(total_elems, 4096),)](
            y2_silu, residual, y_out,
            N, C, H, W,
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
            y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
            num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
