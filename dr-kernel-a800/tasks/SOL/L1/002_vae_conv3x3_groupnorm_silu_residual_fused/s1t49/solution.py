import torch
import triton
import triton.language as tl

@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,          # *float32, input (B, Cin, H, W)
    w_ptr,          # *float32, weight (Cin, Cout, 3, 3)
    out_ptr,        # *float32, output (B, Cout, H, W)
    B: tl.constexpr, Cin: tl.constexpr, Cout: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    eps=0.0,         # unused, for signature compatibility
    BLOCK_IN: tl.constexpr = 32,
):
    # One program computes one output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(axis=0)
    total = B * Cout * H * W
    # Map pid -> (n, c_out, h_out, w_out)
    n = pid // (Cout * H * W)
    tmp = pid % (Cout * H * W)
    c_out = tmp // (H * W)
    tmp2 = tmp % (H * W)
    h_out = tmp2 // W
    w_out = tmp2 % W

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for ci_start in range(0, Cin, BLOCK_IN):
        ci_range = ci_start + tl.arange(0, BLOCK_IN)
        mask_ci = ci_range < Cin
        # Loop over 3x3 window with masks for padding
        for kh in range(3):
            for kw in range(3):
                # compute input coordinates with padding
                h_in = h_out + kh - 1
                w_in = w_out + kw - 1
                # bounds check
                in_h_ok = (h_in >= 0) & (h_in < H)
                in_w_ok = (w_in >= 0) & (w_in < W)
                # compute input offsets for vector ci
                x_off = (n * x_stride_b) + (ci_range[None, :] * x_stride_c) + (h_in * x_stride_h) + (w_in * x_stride_w)
                mask_load = mask_ci[None, :] & in_h_ok & in_w_ok
                x_val = tl.load(x_ptr + x_off, mask=mask_load, other=0.0)  # shape (1, BLOCK_IN)
                # compute weight offsets for this (ci, c_out, kh, kw)
                w_off = (ci_range[:, None] * w_stride_ci) + (c_out * w_stride_co) + (kh * w_stride_kh) + (kw * w_stride_kw)
                w_val = tl.load(w_ptr + w_off, mask=mask_ci[:, None], other=0.0)  # shape (BLOCK_IN, 1)
                # outer product accumulate
                acc += tl.sum(x_val * w_val, axis=1)  # sum over BLOCK_IN -> scalar

    # Store output
    out_off = (n * out_stride_b) + (c_out * out_stride_c) + (h_out * out_stride_h) + (w_out * out_stride_w)
    tl.store(out_ptr + out_off, acc)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,         # *float32, input (B, C, H, W)
    y_ptr,         # *float32, output (B, C, H, W)
    gamma_ptr,     # *float32, per-channel scale (C,)
    beta_ptr,      # *float32, per-channel bias (C,)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    eps: tl.constexpr,
    group_size: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(axis=0)
    n = pid // num_groups
    g = pid % num_groups

    # Compute sum and sumsq over this group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Pass 1: accumulate
    for c in range(0, C):
        if (c % num_groups) == g:
            for h in range(0, H):
                for w in range(0, W):
                    x_off = n * x_stride_b + c * x_stride_c + h * x_stride_h + w * x_stride_w
                    x_val = tl.load(x_ptr + x_off)
                    sum_val += x_val
                    sum_sq += x_val * x_val

    # Compute mean and variance
    M = group_size * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, store
    for c in range(0, C):
        if (c % num_groups) == g:
            gamma = tl.load(gamma_ptr + c)
            beta = tl.load(beta_ptr + c)
            for h in range(0, H):
                for w in range(0, W):
                    x_off = n * x_stride_b + c * x_stride_c + h * x_stride_h + w * x_stride_w
                    y_off = n * y_stride_b + c * y_stride_c + h * y_stride_h + w * y_stride_w
                    x_val = tl.load(x_ptr + x_off)
                    y_val = (x_val - mean) * inv_std
                    y_val = y_val * gamma + beta
                    tl.store(y_ptr + y_off, y_val)


@triton.jit
def silu_kernel(x_ptr, y_ptr, total, BLOCK: tl.constexpr):
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, total, BLOCK: tl.constexpr):
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Triton-only fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add residual (original x)
        Shapes:
          x: (B, C, H, W)
          conv weights: (Cin, Cout, 3, 3)
          norm weights/bias: (C,)
        """
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        device = x.device

        # Cast to float32 and make contiguous
        x_f32 = x.contiguous().to(torch.float32)
        B, Cin, H, W = x_f32.shape
        conv1_weight_f32 = conv1_weight.contiguous().to(torch.float32)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        conv2_weight_f32 = conv2_weight.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        Cout = conv1_weight_f32.shape[1]
        # First conv
        out1 = torch.empty((B, Cin, H, W), dtype=torch.float32, device=device)
        grid_conv1 = (B * Cin * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x_f32, conv1_weight_f32, out1,
            B, Cin, Cin,  # here Cin is input channels, Cin is also output channels in original, but we need Cin as input for conv1
            H, W,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
            conv1_weight_f32.stride(0), conv1_weight_f32.stride(1), conv1_weight_f32.stride(2), conv1_weight_f32.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK_IN=32,
            num_warps=4,
        )

        # First GroupNorm
        num_groups = 32
        assert Cin % num_groups == 0, "Cin must be divisible by num_groups"
        group_size = Cin // num_groups
        y1 = torch.empty((B, Cin, H, W), dtype=torch.float32, device=device)
        grid_gn1 = (B * num_groups,)
        group_norm_affine_kernel[grid_gn1](
            out1, y1, norm1_weight_f32, norm1_bias_f32,
            B, Cin, H, W, num_groups,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            eps=self.eps,
            group_size=group_size,
            num_warps=4,
        )

        # SiLU on y1
        y1_silu = torch.empty_like(y1, device=device, dtype=torch.float32)
        total1 = y1.numel()
        BLOCK_SILU = 1024
        grid_silu1 = (triton.cdiv(total1, BLOCK_SILU),)
        silu_kernel[grid_silu1](y1, y1_silu, total1, BLOCK=BLOCK_SILU, num_warps=4)

        # Second conv
        out2 = torch.empty((B, Cin, H, W), dtype=torch.float32, device=device)  # conv2 expects (B, Cin, H, W) input, outputs same shape
        grid_conv2 = (B * Cin * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight_f32, out2,
            B, Cin, Cin, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight_f32.stride(0), conv2_weight_f32.stride(1), conv2_weight_f32.stride(2), conv2_weight_f32.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK_IN=32,
            num_warps=4,
        )

        # Second GroupNorm
        assert Cin % num_groups == 0, "Cin must be divisible by num_groups"
        group_size2 = Cin // num_groups
        y2 = torch.empty((B, Cin, H, W), dtype=torch.float32, device=device)
        grid_gn2 = (B * num_groups,)
        group_norm_affine_kernel[grid_gn2](
            out2, y2, norm2_weight_f32, norm2_bias_f32,
            B, Cin, H, W, num_groups,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            eps=self.eps,
            group_size=group_size2,
            num_warps=4,
        )

        # SiLU on y2
        y2_silu = torch.empty_like(y2, device=device, dtype=torch.float32)
        total2 = y2.numel()
        grid_silu2 = (triton.cdiv(total2, BLOCK_SILU),)
        silu_kernel[grid_silu2](y2, y2_silu, total2, BLOCK=BLOCK_SILU, num_warps=4)

        # Add residual: original input x (cast to fp32 and made contiguous) added to final output
        x_residual = x_f32.contiguous()
        out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total_final = y2_silu.numel()
        grid_add = (triton.cdiv(total_final, BLOCK_SILU),)
        add_residual_kernel[grid_add](x_residual, y2_silu, out, total_final, BLOCK=BLOCK_SILU, num_warps=4)

        return out


# Optional local test
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 16, 64, 64, 64
    conv1_weight = torch.randn(C, C, 3, 3, device=device, dtype=torch.float32)
    norm1_weight = torch.randn(C, device=device, dtype=torch.float32)
    norm1_bias = torch.randn(C, device=device, dtype=torch.float32)
    conv2_weight = torch.randn(C, C, 3, 3, device=device, dtype=torch.float32)
    norm2_weight = torch.randn(C, device=device, dtype=torch.float32)
    norm2_bias = torch.randn(C, device=device, dtype=torch.float32)
    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)

    model = ModelNew(eps=1e-5).to(device)
    with torch.no_grad():
        y = model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)
    print(y.shape)  # (B, C, H, W)


def run(*args):
    return ModelNew()(*args)
