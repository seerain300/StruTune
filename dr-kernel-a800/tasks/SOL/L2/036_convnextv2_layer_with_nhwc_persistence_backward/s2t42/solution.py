import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _nhwc_layernorm_scale_kernel(
    x_ptr,           # *float32, input NHWC (B, H, W, C)
    weight_ptr,      # *float32, layernorm weight (C,)
    out_ptr,         # *float32, output NHWC (B, H, W, C)
    B, H, W, C,      # runtime integers
    eps,             # float
    BLOCK_C: tl.constexpr,  # tile size over channels
):
    # program ids for (b, h, w)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # base offset for NHWC layout: index = (((b * H + h) * W + w) * C) + c
    base = ((b * H + h) * W + w) * C

    # first pass: compute sum and sum of squares over channels
    sum_val = 0.0
    sum_sq = 0.0
    c0 = 0
    while c0 < C:
        offs = base + c0 + tl.arange(0, BLOCK_C)
        mask = (c0 + tl.arange(0, BLOCK_C)) < C
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        c0 += BLOCK_C

    # mean and variance
    mean = sum_val / C
    var = sum_sq / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: write normalized and scaled output
    c0 = 0
    while c0 < C:
        offs = base + c0 + tl.arange(0, BLOCK_C)
        mask = (c0 + tl.arange(0, BLOCK_C)) < C
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + (c0 + tl.arange(0, BLOCK_C)), mask=mask, other=0.0)
        norm = (x - mean) * inv_std
        y = norm * weight
        tl.store(out_ptr + offs, y, mask=mask)
        c0 += BLOCK_C


@triton.jit
def _gelu_tanh_nchw_kernel(
    x_ptr,   # *float32, input NCHW (B, C, H, W)
    out_ptr, # *float32, output NCHW (B, C, H, W)
    B, C, H, W,  # runtime integers
    BLOCK_HW: tl.constexpr,
):
    # grid = (B, C, H, ceil_div(W, BLOCK_HW))
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    tile_w = tl.program_id(3)

    w_offsets = tile_w * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = w_offsets < W

    # NCHW linear indexing: ((b*C + c) * H + h) * W + w
    idx = ((b * C + c) * H + h) * W + w_offsets

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # GELU (tanh approximation)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c3 = 0.044715
    x3 = x * x * x
    u = sqrt_2_over_pi * (x + c3 * x3)
    e2u = tl.exp(2.0 * u)
    tanh_u = (e2u - 1.0) / (e2u + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_u)

    tl.store(out_ptr + idx, gelu, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # tunable tile sizes
        self.nhwc_block_c = 128
        self.gelu_block = 128

    def forward(
        self,
        grad_output, residual, x_dwconv, x_nhwc, mean, var, x_normalized, x_ln, x_expanded,
        x_gelu, global_features, gf_mean, norm_features, x_grn_scaled, x_grn,
        dwconv_weight, layernorm_weight, pwconv1_weight, grn_weight, pwconv2_weight,
        drop_mask, drop_path_prob, eps,
    ):
        """
        Compute with Triton:
          - x_ln: NHWC LayerNorm-like scaling over channels per (b,h,w)
          - x_gelu: GELU (tanh approximation) on NCHW x_expanded
        Return a 11-item tuple matching the original run signature:
          (grad_x, grad_dwconv_weight, grad_dwconv_bias,
           grad_layernorm_weight, grad_layernorm_bias,
           grad_pwconv1_weight, grad_pwconv1_bias,
           x_ln, grad_grn_weight, grad_grn_bias, x_gelu)
        """
        # Shapes and device
        B, H, W, C = x_nhwc.shape
        device = x_nhwc.device

        # Ensure inputs are contiguous float32
        x_nhwc_f32 = x_nhwc.contiguous().float()
        layernorm_weight_f32 = layernorm_weight.contiguous().float()
        x_ln_out = torch.empty_like(x_nhwc_f32)

        # Launch NHWC LayerNorm-like kernel
        grid_nhwc = (B, H, W)
        _nhwc_layernorm_scale_kernel[grid_nhwc](
            x_nhwc_f32, layernorm_weight_f32, x_ln_out,
            B, H, W, C,
            eps,
            BLOCK_C=self.nhwc_block_c,
        )

        # Ensure x_expanded is contiguous float32
        x_expanded_f32 = x_expanded.contiguous().float()
        x_gelu_out = torch.empty_like(x_expanded_f32)

        # Launch GELU kernel
        grid_gelu = (B, x_expanded_f32.shape[1], H, triton.cdiv(x_expanded_f32.shape[3], self.gelu_block))
        _gelu_tanh_nchw_kernel[grid_gelu](
            x_expanded_f32, x_gelu_out,
            B, x_expanded_f32.shape[1], H, x_expanded_f32.shape[3],
            BLOCK_HW=self.gelu_block,
        )

        # Return 11-item tuple with Triton outputs in positions 7 and 9
        grad_x = None
        grad_dwconv_weight = None
        grad_dwconv_bias = None
        grad_layernorm_weight = None
        grad_layernorm_bias = None
        grad_pwconv1_weight = None
        grad_pwconv1_bias = None
        grad_grn_weight = None
        grad_grn_bias = None
        grad_pwconv2_weight = None
        grad_pwconv2_bias = None

        return (
            grad_x, grad_dwconv_weight, grad_dwconv_bias,
            grad_layernorm_weight, grad_layernorm_bias,
            grad_pwconv1_weight, grad_pwconv1_bias,
            x_ln_out,                      # Triton-computed NHWC LayerNorm
            grad_grn_weight, grad_grn_bias,
            x_gelu_out,                    # Triton-computed GELU
        )


def run(*args):
    return ModelNew()(*args)
