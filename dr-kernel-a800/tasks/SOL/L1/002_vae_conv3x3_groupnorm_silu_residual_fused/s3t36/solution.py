import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H*W). Each program computes one output element y[b, oc, h, w].
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    sp = tl.program_id(2)  # linear index over H*W
    h = sp // W
    w = sp % W

    # scalar accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels (scalar), padding=1, stride=1
    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = h + kh - 1  # -1, 0, 1 (safe due to padding)
            for kw in range(0, 3):
                iw = w + kw - 1  # -1, 0, 1

                # input pointer (padding handled via mask in PyTorch semantics)
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + ih * x_stride_h \
                         + iw * x_stride_w
                # mask ensures we don't read invalid when ih/iw out of bounds, but here we rely on padding logic.
                # Triton load with mask is better; implement masked load properly:
                x_in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_val = tl.load(x_ptrs, mask=x_in_bounds, other=0.0)

                # weight pointer
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_val = tl.load(w_ptrs)

                acc += x_val * w_val

    # store result
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0.
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # first pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_vals + y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias):
        # Ensure CUDA and contiguous, use float32 for stable math
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be CUDA."
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        B, C, H, W = x.shape
        assert conv1_weight.shape == (C, C, 3, 3) and conv2_weight.shape == (C, C, 3, 3), "Weight shapes must be (C, C, 3, 3)."
        num_groups = 32
        assert C % num_groups == 0, "C must be divisible by num_groups (32)."

        # 1) First conv
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C, H * W)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # 2) GroupNorm + affine + SiLU for first block
        y1g = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_g1 = (B, num_groups)
        group_norm_affine_silu[grid_g1](
            y1, norm1_weight, norm1_bias, y1g,
            B, C, H, W, num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1g.stride(0), y1g.stride(1), y1g.stride(2), y1g.stride(3),
            BLOCK=1024,
        )

        # 3) Second conv
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C, H * W)
        conv3x3_triton[grid2](
            y1g, conv2_weight, y2,
            B, C, H, W, C,
            y1g.stride(0), y1g.stride(1), y1g.stride(2), y1g.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        )

        # 4) GroupNorm + affine + SiLU for second block
        y2g = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_g2 = (B, num_groups)
        group_norm_affine_silu[grid_g2](
            y2, norm2_weight, norm2_bias, y2g,
            B, C, H, W, num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2g.stride(0), y2g.stride(1), y2g.stride(2), y2g.stride(3),
            BLOCK=1024,
        )

        # 5) Residual add: out = y2g + x
        N = B * C * H * W
        out_flat = torch.empty(N, device=x.device, dtype=torch.float32)
        y2g_flat = y2g.view(-1)
        x_flat = x.view(-1)
        grid_add = (triton.cdiv(N, 2048),)
        add_residual_kernel[grid_add](
            out_flat, y2g_flat, x_flat, N, BLOCK=2048
        )
        out = out_flat.view(B, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
