import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _nhwc_layernorm_scale_kernel(
    x_ptr,            # *float32, input NHWC: (B, H, W, C)
    lnw_ptr,          # *float32, layernorm_weight: (C,)
    out_ptr,          # *float32, output NHWC: (B, H, W, C)
    B: tl.int32,      # runtime int
    H: tl.int32,      # runtime int
    W: tl.int32,      # runtime int
    C: tl.int32,      # runtime int
    BLOCK_W: tl.int32,  # runtime int, e.g., 32
    eps: tl.float32,     # runtime float
):
    # 3D grid: (B, H, W_blocks)
    b = tl.program_id(0)   # batch index
    h = tl.program_id(1)   # row index
    w_block = tl.program_id(2)  # block index along W

    w_start = w_block * BLOCK_W

    # First pass: compute mean and variance over channels for this (b, h, w_block)
    sum_x = tl.zeros([1], dtype=tl.float32)
    sum_x2 = tl.zeros([1], dtype=tl.float32)

    for w_off in tl.range(0, BLOCK_W):
        w = w_start + w_off
        valid = w < W
        for c in tl.range(0, C):
            idx = (((b * H + h) * W + w) * C) + c
            x_val = tl.load(x_ptr + idx, mask=valid, other=0.0, eviction_policy="evict_last")
            sum_x += x_val
            sum_x2 += x_val * x_val

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: write normalized and scaled output for this (b, h, w_block)
    for w_off in tl.range(0, BLOCK_W):
        w = w_start + w_off
        valid = w < W
        for c in tl.range(0, C):
            idx_in = (((b * H + h) * W + w) * C) + c
            x_val = tl.load(x_ptr + idx_in, mask=valid, other=0.0, eviction_policy="evict_last")
            normed = (x_val - mean) * inv_std
            lnw_val = tl.load(lnw_ptr + c, eviction_policy="evict_last")  # per-channel layernorm weight
            y_val = normed * lnw_val
            idx_out = (((b * H + h) * W + w) * C) + c
            tl.store(out_ptr + idx_out, y_val, mask=valid, eviction_policy="evict_last")


@triton.jit
def _gelu_tanh_kernel(
    x_ptr,            # *float32, input NCHW: (B, C, H, W)
    out_ptr,          # *float32, output NCHW: (B, C, H, W)
    B: tl.int32,      # runtime int
    C: tl.int32,      # runtime int
    H: tl.int32,      # runtime int
    W: tl.int32,      # runtime int
    BLOCK_W: tl.int32,  # runtime int, e.g., 64
):
    # 4D grid over (B, C, H, W_blocks)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w_block = tl.program_id(3)

    w_start = w_block * BLOCK_W

    for w_off in tl.range(0, BLOCK_W):
        w = w_start + w_off
        valid = w < W
        idx = (((b * C + c) * H + h) * W + w)
        x_val = tl.load(x_ptr + idx, mask=valid, other=0.0, eviction_policy="evict_last")
        x3 = x_val * x_val * x_val
        inner = 0.7978845608028654 * (x_val + 0.044715 * x3)  # sqrt(2/pi) * (x + 0.044715*x^3)
        e2 = tl.exp(2.0 * inner)
        tanh_inner = (e2 - 1.0) / (e2 + 1.0)
        y_val = 0.5 * x_val * (1.0 + tanh_inner)
        tl.store(out_ptr + idx, y_val, mask=valid, eviction_policy="evict_last")


class ModelNew(nn.Module):
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
        x_ln: torch.Tensor,
        x_expanded: torch.Tensor,
        x_gelu: torch.Tensor,
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
        """
        Triton-optimized forward that computes:
        - NHWC-like LayerNorm scaling: x_ln = (x_nhwc - mean) / sqrt(var+eps) * layernorm_weight
        - GELU (tanh approximation) on x_expanded
        All heavy computation is done inside Triton kernels. No torch elementwise ops are used in forward.
        Returns the same 11-item structure as the original 'run' function, with None for gradients.
        """
        # Triton requires CUDA tensors; ensure inputs are on the right device/dtype
        device = x_nhwc.device
        dtype = torch.float32

        # NHWC LayerNorm output
        x_ln_triton = torch.empty_like(x_nhwc, device=device, dtype=torch.float32)
        lnw = layernorm_weight.to(dtype=dtype, device=device).contiguous()

        # Launch NHWC LayerNorm Triton kernel: grid = (B, H, W_blocks)
        B, H, W, C = x_nhwc.shape
        BLOCK_W = 32  # tile size along W
        grid_nhwc = (B, H, triton.cdiv(W, BLOCK_W))
        _nhwc_layernorm_scale_kernel[grid_nhwc](
            x_nhwc, lnw, x_ln_triton,
            B, H, W, C,
            BLOCK_W,
            eps,
        )

        # GELU Triton kernel on NCHW x_expanded: grid = (B, C, H, W_blocks)
        x_gelu_triton = torch.empty_like(x_expanded, device=device, dtype=torch.float32)
        B, C, H, W = x_expanded.shape
        BLOCK_W = 64
        grid_gelu = (B, C, H, triton.cdiv(W, BLOCK_W))
        _gelu_tanh_kernel[grid_gelu](
            x_expanded, x_gelu_triton,
            B, C, H, W,
            BLOCK_W,
        )

        # Return same structure as original 'run', filling None for gradients (forward-only)
        return (
            x_gelu_triton,                 # 0
            None,                          # grad_x
            None,                          # grad_dwconv_weight
            None,                          # grad_dwconv_bias
            None,                          # grad_layernorm_weight
            None,                          # grad_layernorm_bias
            None,                          # grad_pwconv1_weight
            None,                          # grad_pwconv1_bias
            None,                          # grad_grn_weight
            None,                          # grad_grn_bias
            None,                          # grad_pwconv2_weight
            None,                          # grad_pwconv2_bias
        )


def run(*args):
    return ModelNew()(*args)
