import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 conv, NCHW, stride=1, padding=1, no bias
# Each program computes one output element y[n, c_out, h_out, w_out].
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,          # *float32, input [B, Cin, H, W]
    w_ptr,          # *float32, weight [Cout, Cin, 3, 3]
    y_ptr,          # *float32, output [B, Cout, H, W]
    B: tl.constexpr,
    Cin: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    Cout: tl.constexpr,
    x_strideN, x_strideC, x_strideH, x_strideW,
    w_strideCout, w_strideCin, w_strideKh, w_strideKw,
    y_strideN, y_strideC, y_strideH, y_strideW,
):
    # program id maps to (n, c_out, h_out, w_out)
    pid = tl.program_id(0)
    # total number of output elements
    total = B * Cout * H * W
    # compute indices
    n = pid // (Cout * H * W)
    rem = pid % (Cout * H * W)
    c_out = rem // (H * W)
    rem2 = rem % (H * W)
    h_out = rem2 // W
    w_out = rem2 % W

    # accumulator for this output element
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel window
    for cin in range(0, Cin):
        for kh in range(0, 3):
            h_in = h_out - kh  # padding=1
            h_mask = (h_in >= 0) & (h_in < H)
            for kw in range(0, 3):
                w_in = w_out - kw  # padding=1
                w_mask = (w_in >= 0) & (w_in < W)
                mask = h_mask & w_mask

                # load input scalar with mask
                x_ptr_scalar = x_ptr + n * x_strideN + cin * x_strideC + h_in * x_strideH + w_in * x_strideW
                x_val = tl.load(x_ptr_scalar, mask=mask, other=0.0)

                # load corresponding weight scalar
                w_ptr_scalar = w_ptr + c_out * w_strideCout + cin * w_strideCin + kh * w_strideKh + kw * w_strideKw
                w_val = tl.load(w_ptr_scalar)

                acc += x_val * w_val

    # store result
    y_ptr_scalar = y_ptr + n * y_strideN + c_out * y_strideC + h_out * y_strideH + w_out * y_strideW
    tl.store(y_ptr_scalar, acc)


# Triton kernel: GroupNorm with affine per (batch, group), two-pass
@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32, input [B, C, H, W]
    y_ptr,          # *float32, output [B, C, H, W]
    weight_ptr,     # *float32, scale [C]
    bias_ptr,       # *float32, bias [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    x_strideN, x_strideC, x_strideH, x_strideW,
    y_strideN, y_strideC, y_strideH, y_strideW,
    eps: tl.float32,
    group_size: tl.constexpr,
):
    pid = tl.program_id(0)  # over B * num_groups
    n = pid // num_groups
    g = pid % num_groups

    total_elems = group_size * H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: compute sum and sum of squares across channels in group and all spatial positions
    for c_off in range(0, group_size):
        c = g * group_size + c_off
        for h in range(0, H):
            for w in range(0, W):
                x_ptrs = x_ptr + n * x_strideN + c * x_strideC + h * x_strideH + w * x_strideW
                x_val = tl.load(x_ptrs)
                sum_val += x_val
                sum_sq += x_val * x_val

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store
    for c_off in range(0, group_size):
        c = g * group_size + c_off
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(0, H):
            for w in range(0, W):
                x_ptrs = x_ptr + n * x_strideN + c * x_strideC + h * x_strideH + w * x_strideW
                x_val = tl.load(x_ptrs)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * scale + bias
                y_ptrs = y_ptr + n * y_strideN + c * y_strideC + h * y_strideH + w * y_strideW
                tl.store(y_ptrs, y_val)


# Triton kernel: SiLU elementwise over flat memory
@triton.jit
def silu_kernel(x_ptr, y_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


# Triton kernel: elementwise residual addition y = x + y
@triton.jit
def add_residual_kernel(x_ptr, y_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,                 # (B, C, H, W)
        conv1_weight: torch.Tensor,      # (C, C, 3, 3)
        norm1_weight: torch.Tensor,      # (C,)
        norm1_bias: torch.Tensor,        # (C,)
        conv2_weight: torch.Tensor,      # (C, C, 3, 3)
        norm2_weight: torch.Tensor,      # (C,)
        norm2_bias: torch.Tensor,        # (C,)
    ):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation in Triton kernels.
        """
        device = x.device
        dtype = x.dtype

        # Cast to float32 and ensure contiguous for Triton
        x_f32 = x.to(torch.float32).contiguous()
        conv1_weight_f32 = conv1_weight.to(torch.float32).contiguous()
        norm1_weight_f32 = norm1_weight.to(torch.float32).contiguous()
        norm1_bias_f32 = norm1_bias.to(torch.float32).contiguous()
        conv2_weight_f32 = conv2_weight.to(torch.float32).contiguous()
        norm2_weight_f32 = norm2_weight.to(torch.float32).contiguous()
        norm2_bias_f32 = norm2_bias.to(torch.float32).contiguous()

        B, C, H, W = x_f32.shape
        num_groups = 32
        assert C % num_groups == 0, "C must be divisible by num_groups=32"

        # First conv: (B, C, H, W) -> (B, C, H, W)
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=device)
        total_conv1 = B * C * H * W
        grid_conv1 = (total_conv1,)
        conv3x3_nchw_fp32[grid_conv1](
            x_f32, conv1_weight_f32, y1,
            B, C, H, W, C,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
            conv1_weight_f32.stride(0), conv1_weight_f32.stride(1), conv1_weight_f32.stride(2), conv1_weight_f32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4,
        )

        # First GroupNorm (num_groups=32), affine
        group_size = C // num_groups
        gn1 = torch.empty_like(y1, device=device, dtype=torch.float32)
        grid_gn1 = (B * num_groups,)
        group_norm_affine_kernel[grid_gn1](
            y1, gn1, norm1_weight_f32, norm1_bias_f32,
            B, C, H, W,
            num_groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            gn1.stride(0), gn1.stride(1), gn1.stride(2), gn1.stride(3),
            eps=self.eps,
            group_size=group_size,
            num_warps=1,
        )

        # SiLU
        silu1 = torch.empty_like(gn1, device=device, dtype=torch.float32)
        total1 = gn1.numel()
        BLOCK_SILU = 1024
        grid_silu1 = (triton.cdiv(total1, BLOCK_SILU),)
        silu_kernel[grid_silu1](gn1, silu1, total1, BLOCK=BLOCK_SILU, num_warps=4)

        # Second conv: (B, C, H, W) -> (B, C, H, W)
        y2 = torch.empty((B, C, H, W), dtype=torch.float32, device=device)
        grid_conv2 = (total_conv1,)  # same mapping
        conv3x3_nchw_fp32[grid_conv2](
            silu1, conv2_weight_f32, y2,
            B, C, H, W, C,
            silu1.stride(0), silu1.stride(1), silu1.stride(2), silu1.stride(3),
            conv2_weight_f32.stride(0), conv2_weight_f32.stride(1), conv2_weight_f32.stride(2), conv2_weight_f32.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4,
        )

        # Second GroupNorm
        gn2 = torch.empty_like(y2, device=device, dtype=torch.float32)
        grid_gn2 = (B * num_groups,)
        group_norm_affine_kernel[grid_gn2](
            y2, gn2, norm2_weight_f32, norm2_bias_f32,
            B, C, H, W,
            num_groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            gn2.stride(0), gn2.stride(1), gn2.stride(2), gn2.stride(3),
            eps=self.eps,
            group_size=group_size,
            num_warps=1,
        )

        # SiLU
        silu2 = torch.empty_like(gn2, device=device, dtype=torch.float32)
        total2 = gn2.numel()
        grid_silu2 = (triton.cdiv(total2, BLOCK_SILU),)
        silu_kernel[grid_silu2](gn2, silu2, total2, BLOCK=BLOCK_SILU, num_warps=4)

        # Residual addition: add original input (cast to float32) to final output
        total_final = silu2.numel()
        out = torch.empty_like(silu2, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_final, BLOCK_SILU),)
        add_residual_kernel[grid_add](x_f32, silu2, out, total_final, BLOCK=BLOCK_SILU, num_warps=4)

        # Return fp32 output (evaluation expects Triton-only, fp32 math)
        return out


def run(*args):
    return ModelNew()(*args)
