import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _nhwc_layernorm_scale_kernel(
    x_nhwc_ptr,      # *float32, shape (B, H, W, C)
    ln_weight_ptr,   # *float32, shape (C,)
    out_ptr,         # *float32, shape (B, H, W, C)
    B: tl.int32,
    H: tl.int32,
    W: tl.int32,
    C: tl.int32,
    eps: tl.float32,
):
    # program ids: each program handles one (b, h) row across W
    b = tl.program_id(0)
    h = tl.program_id(1)
    if b >= B or h >= H:
        return

    # Compute mean and variance across channels C for all W positions
    # Initialize scalar accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    # Loop over channels
    for c in range(0, C):
        # Accumulate sum and sum of squares over W
        for w_pos in range(0, W):
            offs = (((b * H) + h) * W + w_pos) * C + c
            x_val = tl.load(x_nhwc_ptr + offs)
            sum_x += x_val
            sum_x2 += x_val * x_val
    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Write normalized and scaled output for each (w, c)
    for w_pos in range(0, W):
        for c in range(0, C):
            offs = (((b * H) + h) * W + w_pos) * C + c
            x_val = tl.load(x_nhwc_ptr + offs)
            norm = (x_val - mean) * inv_std
            weight = tl.load(ln_weight_ptr + c)
            out_val = norm * weight
            tl.store(out_ptr + offs, out_val)


@triton.jit
def _gelu_tanh_kernel(
    x_in_ptr,        # *float32, shape (B, C, H, W)
    out_ptr,         # *float32, shape (B, C, H, W)
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
):
    # 4D grid over (B, C, H, W)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    if b >= B or c >= C or h >= H or w >= W:
        return

    x = tl.load(x_in_ptr + (((b * C) + c) * H + h) * W + w)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    cdf_coeff = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + cdf_coeff * x3)
    # tanh(inner) via exp: tanh(u) = (e^{2u} - 1) / (e^{2u} + 1)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + (((b * C) + c) * H + h) * W + w, gelu)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Unpack inputs: first 10 are outputs, last 13 are params/masks
        grad_output = args[0] if len(args) > 0 else None
        residual = args[1] if len(args) > 1 else None
        x_dwconv = args[2] if len(args) > 2 else None
        x_nhwc = args[3] if len(args) > 3 else None
        mean = args[4] if len(args) > 4 else None
        var = args[5] if len(args) > 5 else None
        x_normalized = args[6] if len(args) > 6 else None
        x_ln = args[7] if len(args) > 7 else None
        x_expanded = args[8] if len(args) > 8 else None
        x_gelu = args[9] if len(args) > 9 else None
        global_features = args[10] if len(args) > 10 else None
        gf_mean = args[11] if len(args) > 11 else None
        norm_features = args[12] if len(args) > 12 else None
        x_grn_scaled = args[13] if len(args) > 13 else None
        x_grn = args[14] if len(args) > 14 else None
        dwconv_weight = args[15] if len(args) > 15 else None
        layernorm_weight = args[16] if len(args) > 16 else None
        pwconv1_weight = args[17] if len(args) > 17 else None
        grn_weight = args[18] if len(args) > 18 else None
        pwconv2_weight = args[19] if len(args) > 19 else None
        drop_mask = args[20] if len(args) > 20 else None
        drop_path_prob = args[21] if len(args) > 21 else 0.1
        eps = args[22] if len(args) > 22 else 1e-6

        # Triton: NHWC LayerNorm-like scaling
        if x_nhwc is not None:
            x_nhwc = x_nhwc.contiguous().to(torch.float32)
            B, H, W, C = x_nhwc.shape
            x_ln_out = torch.empty_like(x_nhwc, dtype=torch.float32, device=x_nhwc.device)
            grid = (B, H)
            _nhwc_layernorm_scale_kernel[grid](
                x_nhwc, layernorm_weight, x_ln_out,
                B, H, W, C, eps,
                num_warps=4,
            )
            x_ln = x_ln_out
        else:
            x_ln = None

        # Triton: GELU on NCHW
        if x_expanded is not None:
            x_expanded = x_expanded.contiguous().to(torch.float32)
            B, C, H, W = x_expanded.shape
            x_gelu_out = torch.empty_like(x_expanded, dtype=torch.float32, device=x_expanded.device)
            grid = (B, C, H, W)
            _gelu_tanh_kernel[grid](
                x_expanded, x_gelu_out,
                B, C, H, W,
                num_warps=4,
            )
            x_gelu = x_gelu_out
        else:
            x_gelu = None

        return (
            grad_output,
            residual,
            x_dwconv,
            x_nhwc,
            mean,
            var,
            x_normalized,
            x_ln,
            x_expanded,
            x_gelu,
            global_features,
            gf_mean,
            norm_features,
            x_grn_scaled,
            x_grn,
            dwconv_weight,
            layernorm_weight,
            pwconv1_weight,
            grn_weight,
            pwconv2_weight,
            drop_mask,
            drop_path_prob,
            eps,
        )


def run(*args):
    return ModelNew()(*args)
