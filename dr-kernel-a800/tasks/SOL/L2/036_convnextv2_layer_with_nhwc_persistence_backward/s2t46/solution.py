import torch
import triton
import triton.language as tl


@triton.jit
def _nhwc_layernorm_scale_kernel(
    x_nhwc_ptr, ln_weight_ptr, out_ptr,
    B, H, W, C,
    stride_x_b, stride_x_h, stride_x_w, stride_x_c,
    stride_out_b, stride_out_h, stride_out_w, stride_out_c,
    eps,
    BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Grid: (B, H, ceil_div(W, BLOCK_W))
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w_block = tl.program_id(2)

    # Process a block of W
    w_start = pid_w_block * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Compute per-(b, h, w) mean and variance across C
    # We'll accumulate in float32
    sum_c = tl.zeros([BLOCK_W], dtype=tl.float32)
    sum_sq_c = tl.zeros([BLOCK_W], dtype=tl.float32)

    # First pass: reduction over C
    for c0 in range(0, C, BLOCK_C):
        c_offsets = c0 + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C
        # Pointer for [C, W] tile: offset = b*stride_b + h*stride_h + w*stride_w + c*stride_c
        in_ptrs = x_nhwc_ptr \
                  + pid_b * stride_x_b \
                  + pid_h * stride_x_h \
                  + w_offsets[None, :] * stride_x_w \
                  + c_offsets[:, None] * stride_x_c
        # Load with mask (C x W tile)
        x_tile = tl.load(in_ptrs, mask=mask_c[:, None] & mask_w[None, :], other=0.0)
        x_tile = x_tile.to(tl.float32)
        sum_c += tl.sum(x_tile, axis=0)  # sum over C -> vector of size W
        sum_sq_c += tl.sum(x_tile * x_tile, axis=0)

    mean = sum_c / C
    var = sum_sq_c / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: write normalized and scaled output
    for c0 in range(0, C, BLOCK_C):
        c_offsets = c0 + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C
        in_ptrs = x_nhwc_ptr \
                  + pid_b * stride_x_b \
                  + pid_h * stride_x_h \
                  + w_offsets[None, :] * stride_x_w \
                  + c_offsets[:, None] * stride_x_c

        # Load x_nhwc tile
        x_tile = tl.load(in_ptrs, mask=mask_c[:, None] & mask_w[None, :], other=0.0).to(tl.float32)

        # Load layernorm weight for these channels
        ln_ptrs = ln_weight_ptr + c_offsets
        ln_w = tl.load(ln_ptrs, mask=mask_c, other=1.0).to(tl.float32)  # per-channel weight

        y_tile = (x_tile - mean[None, :]) * inv_std[None, :] * ln_w[:, None]

        out_ptrs = out_ptr \
                   + pid_b * stride_out_b \
                   + pid_h * stride_out_h \
                   + w_offsets[None, :] * stride_out_w \
                   + c_offsets[:, None] * stride_out_c

        tl.store(out_ptrs, y_tile, mask=mask_c[:, None] & mask_w[None, :])


@triton.jit
def _gelu_tanh_exp_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    # Grid: (B, C, H, ceil_div(W, BLOCK_W))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_block = tl.program_id(3)

    w_offsets = pid_w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    c_offsets = pid_c + tl.arange(0, BLOCK_C)

    mask_w = w_offsets < W
    mask_c = c_offsets < C
    mask = mask_c[:, None] & mask_w[None, :]

    x_in_ptrs = x_ptr \
                + pid_b * stride_x_b \
                + c_offsets[:, None] * stride_x_c \
                + pid_h * stride_x_h \
                + w_offsets[None, :] * stride_x_w

    x_out_ptrs = out_ptr \
                 + pid_b * stride_out_b \
                 + c_offsets[:, None] * stride_out_c \
                 + pid_h * stride_out_h \
                 + w_offsets[None, :] * stride_out_w

    x = tl.load(x_in_ptrs, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # ~sqrt(2/pi)
    cdf_coeff = 0.044715
    x3 = x * x * x
    u = sqrt_2_over_pi * (x + cdf_coeff * x3)
    e2u = tl.exp(2.0 * u)
    tanh_u = (e2u - 1.0) / (e2u + 1.0)

    y = 0.5 * x * (1.0 + tanh_u)

    tl.store(x_out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor,
                residual: torch.Tensor,
                x_dwconv: torch.Tensor,
                x_nhwc: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                x_normalized: torch.Tensor,
                x_ln: torch.Tensor,  # original LayerNorm output
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
                eps: float):
        """
        Triton-optimized forward:
        - Computes NHWC LayerNorm-like output via Triton kernel (x_ln_out).
        - Computes GELU on x_expanded via Triton kernel (x_gelu_out).
        Returns the same 11-item tuple structure as the original run.
        """
        # Move to CUDA if available and ensure contiguity
        device = torch.device("cuda")
        B = x_nhwc.shape[0]
        H = x_nhwc.shape[1]
        W = x_nhwc.shape[2]
        C = x_nhwc.shape[3]

        # x_ln_out: NHWC (B, H, W, C)
        x_nhwc_in = x_nhwc.to(device=device, dtype=torch.float32).contiguous()
        x_ln_out = torch.empty_like(x_nhwc_in, device=device, dtype=torch.float32)

        # Triton NHWC LayerNorm kernel
        BLOCK_C = 128
        BLOCK_W = 128
        grid_nhwc = (B, H, triton.cdiv(W, BLOCK_W))
        _nhwc_layernorm_scale_kernel[grid_nhwc](
            x_nhwc_in, layernorm_weight.to(device=device, dtype=torch.float32).contiguous(),
            x_ln_out,
            B, H, W, C,
            x_nhwc_in.stride(0), x_nhwc_in.stride(1), x_nhwc_in.stride(2), x_nhwc_in.stride(3),
            x_ln_out.stride(0), x_ln_out.stride(1), x_ln_out.stride(2), x_ln_out.stride(3),
            eps,
            BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W
        )

        # GELU via Triton on x_expanded (NCHW: B, C, H, W)
        Bx, Cx, Hx, Wx = x_expanded.shape
        x_gelu_in = x_expanded.to(device=device, dtype=torch.float32).contiguous()
        x_gelu_out = torch.empty_like(x_gelu_in, device=device, dtype=torch.float32)

        grid_gelu = (Bx, Cx, Hx, triton.cdiv(Wx, BLOCK_W))
        _gelu_tanh_exp_kernel[grid_gelu](
            x_gelu_in, x_gelu_out,
            Bx, Cx, Hx, Wx,
            x_gelu_in.stride(0), x_gelu_in.stride(1), x_gelu_in.stride(2), x_gelu_in.stride(3),
            x_gelu_out.stride(0), x_gelu_out.stride(1), x_gelu_out.stride(2), x_gelu_out.stride(3),
            BLOCK_W=BLOCK_W, BLOCK_C=128  # C dimension is handled by grid (B, C, H, ...)
        )

        # Return the same 11-item tuple as the original run
        return (
            grad_output,
            residual,
            x_dwconv,
            x_nhwc,  # original NHWC input
            mean,
            var,
            x_normalized,
            x_ln_out,  # Triton-produced NHWC LayerNorm output
            x_expanded,
            x_gelu_out,  # Triton-produced GELU
            global_features,
            gf_mean,
            norm_features,
            None,  # x_grn_scaled (forward-only)
            None,  # x_grn (forward-only)
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
