import torch
import triton
import triton.language as tl


# Triton kernel: NHWC LayerNorm-like scaling
# Input: x_nhwc  [B, H, W, C], float32, contiguous
# Weight: layernorm_weight [C], float32
# Output: x_ln  [B, H, W, C], float32
@triton.jit
def _nhwc_layernorm_scale_kernel(
    x_ptr,           # *const float
    layernorm_ptr,   # *const float
    out_ptr,         # *float
    B, H, W, C, eps,  # runtime ints
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Compute sum and sum of squares over C for this (b, h, w)
    sum_x = 0.0
    sum_x2 = 0.0
    for c0 in range(0, C, BLOCK_C):
        for cc in range(0, BLOCK_C):
            c = c0 + cc
            if c < C:
                idx = (((b * H) + h) * W + w) * C + c
                x = tl.load(x_ptr + idx).to(tl.float32)
                sum_x += x
                sum_x2 += x * x

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Write normalized and scaled output: out = (x - mean) * inv_std * layernorm_weight[c]
    for c0 in range(0, C, BLOCK_C):
        for cc in range(0, BLOCK_C):
            c = c0 + cc
            if c < C:
                idx = (((b * H) + h) * W + w) * C + c
                x = tl.load(x_ptr + idx).to(tl.float32)
                norm = (x - mean) * inv_std
                wgt = tl.load(layernorm_ptr + c).to(tl.float32)
                out = norm * wgt
                tl.store(out_ptr + idx, out)


# Triton kernel: GELU (tanh approximation) on NCHW tensor
# Input: x_expanded [B, C, H, W], float32, contiguous
# Output: x_gelu [B, C, H, W], float32
@triton.jit
def _gelu_tanh_kernel(
    x_ptr,         # *const float
    out_ptr,       # *float
    B, C, H, W,    # runtime ints
    K: tl.constexpr,  # sqrt(2/pi) = 0.7978845608028654
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    idx = (((b * C) + c) * H + h) * W + w
    x = tl.load(x_ptr + idx).to(tl.float32)

    # GELU tanh approximation
    x3 = x * x * x
    inner = K * (x + 0.044715 * x3)
    e2 = tl.exp(2.0 * inner)
    tanh_inner = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + idx, gelu)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        residual: torch.Tensor,
        x_dwconv: torch.Tensor,
        x_nhwc: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        x_normalized: torch.Tensor,
        x_ln: torch.Tensor,   # Output placeholder (will be overwritten by Triton)
        x_expanded: torch.Tensor,
        x_gelu: torch.Tensor,  # Output placeholder (will be overwritten by Triton)
        global_features: torch.Tensor,
        gf_mean: torch.Tensor,
        norm_features: torch.Tensor,
        x_grn_scaled: torch.Tensor,
        x_grn: torch.Tensor,
        dwconv_weight: torch.Tensor,
        layernorm_weight: torch.Tensor,
        pwconv1_weight: torch.Tensor,
        grn_weight: torch.Tensor,
        pwconv2_weight: torch.Tensor,
        drop_mask: torch.Tensor,
        drop_path_prob: float,
        eps: float,
    ):
        # Triton requires CUDA device
        device = x_nhwc.device
        assert device.type == 'cuda', "Triton requires CUDA device"

        # Dimensions
        B = x_nhwc.shape[0]
        H = x_nhwc.shape[1]
        W = x_nhwc.shape[2]
        C = x_nhwc.shape[3]
        C_expanded = x_expanded.shape[1]

        # 1) NHWC LayerNorm-like scaling via Triton
        x_nhwc_f32 = x_nhwc.contiguous().to(torch.float32)
        layernorm_weight_f32 = layernorm_weight.contiguous().to(torch.float32)
        x_ln = torch.empty_like(x_nhwc_f32)

        BLOCK_C = 64  # chunk for channel reduction
        grid = (B, H, W)
        _nhwc_layernorm_scale_kernel[grid](
            x_nhwc_f32, layernorm_weight_f32, x_ln,
            B, H, W, C, eps,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )

        # 2) GELU (tanh approximation) on NCHW via Triton
        x_expanded_f32 = x_expanded.contiguous().to(torch.float32)
        x_gelu = torch.empty_like(x_expanded_f32)

        sqrt_2_over_pi = 0.7978845608028654
        grid_gelu = (B, C_expanded, H, W)
        _gelu_tanh_kernel[grid_gelu](
            x_expanded_f32, x_gelu,
            B, C_expanded, H, W,
            K=sqrt_2_over_pi,
            num_warps=1,
        )

        # Return structure identical to original, with Triton-computed outputs
        return (
            grad_output,                # unchanged
            residual,                   # unchanged
            x_dwconv,                   # unchanged
            x_nhwc,                     # unchanged
            mean,                       # unchanged
            var,                        # unchanged
            x_normalized,               # unchanged
            x_ln,                       # Triton-computed LayerNorm output
            x_expanded,                 # unchanged
            x_gelu,                     # Triton-computed GELU output
            global_features,            # unchanged
            gf_mean,                    # unchanged
            norm_features,              # unchanged
            x_grn_scaled,               # unchanged
            x_grn,                      # unchanged
            dwconv_weight,              # unchanged
            layernorm_weight,           # unchanged
            pwconv1_weight,             # unchanged
            grn_weight,                 # unchanged
            pwconv2_weight,             # unchanged
            drop_mask,                  # unchanged
            drop_path_prob,             # unchanged
            eps,                        # unchanged
        )


def run(*args):
    return ModelNew()(*args)
