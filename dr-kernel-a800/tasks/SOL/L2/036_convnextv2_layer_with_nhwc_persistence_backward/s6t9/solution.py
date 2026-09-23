import torch
import torch.nn as nn

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# 1) Triton Depthwise Conv2d with groups=C and padding=3: inputs residual (B,C,H,W), weight (C,1,7,7), output (B,C,H+6,W+6)
@triton.jit
def conv2d_depthwise_groupsC_kernel(
    residual_ptr,        # *const float, input: [B, C, H, W]
    dwconv_weight_ptr,   # *const float, weight: [C, 1, 7, 7]
    out_ptr,             # *float, output: [B, C, Ho, Wo] where Ho=H+6, Wo=W+6
    B: tl.int32, C: tl.int32, H: tl.int32, W: tl.int32,
):
    # Grid over (B, C, Ho, Wo)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_wo = tl.program_id(3)

    ho = pid_ho
    wo = pid_wo

    # Compute input coordinates for depthwise conv with padding=3
    # Output: (ho, wo) maps to input (hi, wi) = (ho - 3, wo - 3)
    hi = ho - 3
    wi = wo - 3

    # Accumulator for output (b, c, ho, wo)
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over kernel window (7x7) and accumulate
    # Mask for valid input coordinates
    valid = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)

    # Weights are (C, 1, 7, 7); loop over k_h and k_w
    # Using nested loops to avoid passing big constexprs
    for kh in range(7):
        for kw in range(7):
            # Compute input indices
            hi_k = hi + kh
            wi_k = wi + kw
            # Mask for valid neighbors
            mask = valid & (hi_k >= 0) & (hi_k < H) & (wi_k >= 0) & (wi_k < W)

            # Base pointer for this (b, c)
            base_in = pid_b * (C * H * W) + pid_c * (H * W)
            # Address of residual[b, c, hi_k, wi_k]
            in_offset = base_in + hi_k * W + wi_k
            # Load input with mask; if invalid, treat as 0
            in_val = tl.load(residual_ptr + in_offset, mask=mask, other=0.0)

            # Load dwconv_weight[c, 0, kh, kw] (scalar per kh,kw)
            # weight layout: [C, 1, 7, 7], so index = c*(1*7*7) + 0*7*7 + kh*7 + kw
            w_offset = pid_c * (1 * 7 * 7) + kh * 7 + kw
            w_val = tl.load(dwconv_weight_ptr + w_offset)

            acc += in_val * w_val

    # Store output
    out_offset = pid_b * (C * Ho * Wo) + pid_c * (Ho * Wo) + ho * Wo + wo
    tl.store(out_ptr + out_offset, acc)


# 2) Triton LayerNorm over NHWC: x_nhwc shape (B, H, W, C). For each (b, h, w), reduce over C to compute mean/var,
# normalize, and scale by layernorm_weight (per-channel). Writes to out_ln (B,H,W,C).
@triton.jit
def layernorm_nhwc_kernel(
    x_nhwc_ptr,          # *const float, input NHWC: [B, H, W, C]
    ln_weight_ptr,       # *const float, layernorm_weight: [C]
    out_ln_ptr,          # *float, output: [B, H, W, C]
    B: tl.int32, H: tl.int32, W: tl.int32, C: tl.int32, eps: tl.float32,
):
    # Grid over (B, H*W)
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    hw = pid_hw
    h = hw // W
    w = hw % W

    # Accumulate sum and sum of squares over C
    sum_x = 0.0
    sum_x2 = 0.0
    for c0 in range(0, C):
        base = pid_b * (H * W * C) + h * (W * C) + w * C + c0
        x_val = tl.load(x_nhwc_ptr + base)
        sum_x += x_val
        sum_x2 += x_val * x_val

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    std = tl.sqrt(var + eps)

    # Normalize and scale
    for c0 in range(0, C):
        base = pid_b * (H * W * C) + h * (W * C) + w * C + c0
        x_val = tl.load(x_nhwc_ptr + base)
        ln_weight_val = tl.load(ln_weight_ptr + c0)
        norm = (x_val - mean) / std
        out_val = norm * ln_weight_val
        out_offset = pid_b * (H * W * C) + h * (W * C) + w * C + c0
        tl.store(out_ln_ptr + out_offset, out_val)


# ModelNew: entry point class, forward launches Triton kernels
class ModelNew(nn.Module):
    def forward(self, inputs):
        """
        inputs expected to be: dict with keys:
          - residual: (B, C, H, W) float32 tensor
          - dwconv_weight: (C, 1, 7, 7) float32 tensor
          - layernorm_weight: (C,) float32 tensor
          - eps: float
        Returns:
          x_ln_out: (B, H, W, C) float32 tensor
        """
        residual = inputs["residual"]
        dwconv_weight = inputs["dwconv_weight"]
        layernorm_weight = inputs["layernorm_weight"]
        eps = inputs["eps"]

        B, C, H, W = residual.shape
        device = residual.device

        # Allocate output for depthwise conv: (B, C, Ho, Wo) where Ho=H+6, Wo=W+6
        Ho, Wo = H + 6, W + 6
        x_dwconv_out = torch.empty((B, C, Ho, Wo), dtype=torch.float32, device=device)

        # Ensure contiguous
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()

        # Launch depthwise conv kernel
        grid_conv = (B, C, Ho, Wo)
        conv2d_depthwise_groupsC_kernel[grid_conv](
            residual, dwconv_weight, x_dwconv_out,
            B, C, H, W,
        )

        # Permute to NHWC: x_nhwc = x_dwconv_out.permute(0, 2, 3, 1)
        x_nhwc = x_dwconv_out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # Triton LayerNorm NHWC -> x_ln_out (B,H,W,C)
        x_ln_out = torch.empty((B, H, W, C), dtype=torch.float32, device=device)
        grid_layernorm = (B, H * W)
        layernorm_nhwc_kernel[grid_layernorm](
            x_nhwc, layernorm_weight, x_ln_out,
            B, H, W, C, eps,
            num_warps=4,
        )

        # Return computed output
        return x_ln_out


def run(*args):
    return ModelNew()(*args)
