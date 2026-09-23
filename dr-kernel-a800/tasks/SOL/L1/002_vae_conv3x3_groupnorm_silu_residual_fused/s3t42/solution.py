import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C, H, W), w: (C, C, 3, 3), y: (B, C, H, W)
# Grid: (B, C, H, W). Each program handles one output element (b, oc, h, w).
@triton.jit
def conv3x3_peroc_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    h = tl.program_id(2)
    w_out = tl.program_id(3)

    # accumulator for a single output element
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels (C_in == C), 3x3 neighborhood with padding=1
    for ic in range(0, C):
        for kh in range(-1, 2):
            in_h = h + kh
            for kw in range(-1, 2):
                in_w = w_out + kw
                in_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                x_ptrs = x_ptr + b * x_stride_b + ic * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                x_val = tl.load(x_ptrs, mask=in_mask, other=0.0)

                # weight at (ic, oc, kh+1, kw+1)
                w_ptrs = w_ptr + ic * w_stride_cin + oc * w_stride_cout + (kh + 1) * w_stride_kh + (kw + 1) * w_stride_kw
                w_val = tl.load(w_ptrs)
                acc += x_val * w_val

    # store result
    y_ptrs = y_ptr + b * y_stride_b + oc * y_stride_c + h * y_stride_h + w_out * y_stride_w
    tl.store(y_ptrs, acc)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU
# x: (B, C, H, W), y: (B, C, H, W), weight: (C,), bias: (C,)
# Grid: (B, num_groups). Each program handles one (batch, group).
@triton.jit
def group_norm_affine_silu_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for idx in range(0, group_elements):
        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_val = tl.load(x_ptrs)
        sum_val += x_val
        sum_sq += x_val * x_val

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for idx in range(0, group_elements):
        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_val = tl.load(x_ptrs)

        norm_vals = (x_val - mean) * inv_std

        scale = tl.load(weight_ptr + ch)
        bias = tl.load(bias_ptr + ch)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out_vals = y_vals + x_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias):
        # Ensure CUDA and contiguity
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be CUDA."
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        B, C, H, W = x.shape
        assert conv1_weight.shape[0] == C and conv1_weight.shape[1] == C and conv1_weight.shape[2] == 3 and conv1_weight.shape[3] == 3
        assert conv2_weight.shape[0] == C and conv2_weight.shape[1] == C and conv2_weight.shape[2] == 3 and conv2_weight.shape[3] == 3
        assert norm1_weight.shape[0] == C and norm1_bias.shape[0] == C
        assert norm2_weight.shape[0] == C and norm2_bias.shape[0] == C
        assert C % self.num_groups == 0, "C must be divisible by num_groups."

        # conv1 via Triton
        y1 = torch.empty_like(x)
        grid1 = (B, C, H, W)
        conv3x3_peroc_kernel[grid1](
            x, conv1_weight, y1,
            B, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # GroupNorm + affine + SiLU for conv1
        y1_grouped = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_grouped,
            B, C, H, W,
            self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
        )

        # conv2 via Triton
        y2_pre = torch.empty_like(y1_grouped)
        grid2 = (B, C, H, W)
        conv3x3_peroc_kernel[grid2](
            y1_grouped, conv2_weight, y2_pre,
            B, C, H, W,
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )

        # GroupNorm + affine + SiLU for conv2
        y2_grouped = torch.empty_like(y2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn2](
            y2_pre, norm2_weight, norm2_bias, y2_grouped,
            B, C, H, W,
            self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_grouped.stride(0), y2_grouped.stride(1), y2_grouped.stride(2), y2_grouped.stride(3),
        )

        # Final residual add via Triton
        out = torch.empty_like(y2_grouped)
        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](out, y2_grouped, x, N, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
