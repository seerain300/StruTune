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
    Ho: tl.int32, Wo: tl.int32,
    pad_h: tl.int32, pad_w: tl.int32,  # padding=3 => 3
    BLOCK: tl.constexpr,               # tile size for output channels
):
    # Grid: (B, C, Ho, Wo)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_wo = tl.program_id(3)

    # Accumulator for output channel pid_c
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over kernel window (7x7)
    for kh in range(0, 7):
        for kw in range(0, 7):
            h_in = pid_ho + pad_h - kh
            w_in = pid_wo + pad_w - kw
            # Valid input bounds
            valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            if valid:
                # For each group channel c, accumulate weighted input
                # residual[b, c, h_in, w_in]
                base_res = pid_b * C * H * W + c * H * W + h_in * W + w_in
                x_val = tl.load(residual_ptr + base_res)  # scalar
                # dwconv weight for this (c, kh, kw): dwconv_weight[c, 0, kh, kw]
                base_w = c * (1 * 7 * 7) + 0 * (7 * 7) + kh * 7 + kw
                w_val = tl.load(dwconv_weight_ptr + base_w)  # scalar
                acc += x_val * w_val

    # Write output: out[b, c, pid_ho, pid_wo] = acc
    out_index = pid_b * C * Ho * Wo + pid_c * Ho * Wo + pid_ho * Wo + pid_wo
    tl.store(out_ptr + out_index, acc)


# 2) Triton LayerNorm over NHWC: x_nhwc shape (B, H, W, C). For each (b, h, w), reduce over C to compute mean/var,
# normalize, and scale by layernorm_weight (per-channel). Writes to out_ln (B,H,W,C).
@triton.jit
def layernorm_nhwc_kernel(
    x_nhwc_ptr,          # *const float, input NHWC: [B, H, W, C]
    ln_weight_ptr,       # *const float, layernorm_weight: [C]
    out_ln_ptr,          # *float, output: [B, H, W, C]
    B: tl.int32,         # runtime
    H: tl.int32,         # runtime
    W: tl.int32,         # runtime
    C: tl.int32,         # runtime
    eps: tl.float32,     # runtime
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)  # over batch
    pid_hw = tl.program_id(1) # over H*W
    h = pid_hw // W
    w = pid_hw % W

    # Accumulate sum and sum of squares over C in chunks
    sum_x = 0.0
    sum_x2 = 0.0
    for c0 in range(0, C, BLOCK_C):
        c_offsets = c0 + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C
        # Base pointer for (b, h, w, c)
        base = pid_b * (H * W * C) + h * (W * C) + w * C + c_offsets
        x_vals = tl.load(x_nhwc_ptr + base, mask=mask_c, other=0.0)
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and scale
    for c0 in range(0, C, BLOCK_C):
        c_offsets = c0 + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C
        base = pid_b * (H * W * C) + h * (W * C) + w * C + c_offsets
        x_vals = tl.load(x_nhwc_ptr + base, mask=mask_c, other=0.0)
        ln_weight_vals = tl.load(ln_weight_ptr + c_offsets, mask=mask_c, other=1.0)
        y_vals = (x_vals - mean) * inv_std * ln_weight_vals
        out_base = pid_b * (H * W * C) + h * (W * C) + w * C + c_offsets
        tl.store(out_ln_ptr + out_base, y_vals, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, dwconv_weight: torch.Tensor, layernorm_weight: torch.Tensor, B: int, H: int, W: int):
        """
        residual: (B, C, H, W)
        dwconv_weight: (C, 1, 7, 7)
        layernorm_weight: (C,)
        Returns:
          x_ln_out: (B, H, W, C) after LayerNorm on NHWC (permuted from x_dwconv)
        """
        device = residual.device
        if not TRITON_AVAILABLE:
            # Fallback: compute with PyTorch if Triton unavailable (not ideal for eval)
            x_dwconv = F.conv2d(residual, dwconv_weight, padding=3, groups=C)
            x_nhwc = x_dwconv.permute(0, 2, 3, 1)
            eps = 1e-6
            Bx, Hx, Wx, Cx = x_nhwc.shape
            x_ln_out = torch.empty_like(x_nhwc)
            for b in range(Bx):
                for h in range(Hx):
                    for w in range(Wx):
                        x_slice = x_nhwc[b, h, w, :]
                        mean = x_slice.mean()
                        var = x_slice.var()
                        y = (x_slice - mean) / torch.sqrt(var + eps) * layernorm_weight
                        x_ln_out[b, h, w, :] = y
            return x_ln_out

        # Ensure inputs are contiguous
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()
        layernorm_weight = layernorm_weight.contiguous()

        C = residual.shape[1]
        Ho, Wo = H + 6, W + 6  # output of conv with padding=3

        # 1) Triton Depthwise Conv2d (groups=C, padding=3) -> x_dwconv_out (B, C, Ho, Wo)
        x_dwconv_out = torch.empty((B, C, Ho, Wo), dtype=torch.float32, device=device)
        grid = (B, C, Ho, Wo)
        conv2d_depthwise_groupsC_kernel[grid](
            residual, dwconv_weight, x_dwconv_out,
            B, C, H, W, Ho, Wo, 3, 3,
            num_warps=4,
        )

        # 2) Permute to NHWC: x_nhwc = x_dwconv_out.permute(0, 2, 3, 1) -> (B, Ho, Wo, C)
        x_nhwc = x_dwconv_out.permute(0, 2, 3, 1).contiguous()  # (B, H+6, W+6, C)

        # 3) Triton LayerNorm over NHWC -> x_ln_out (B,H,W,C)
        x_ln_out = torch.empty((B, H, W, C), dtype=torch.float32, device=device)
        BLOCK_C = 128  # matches C=128; loop in chunks safely
        grid_layernorm = (B, H * W)
        layernorm_nhwc_kernel[grid_layernorm](
            x_nhwc, layernorm_weight, x_ln_out,
            B, H, W, C, 1e-6,
            BLOCK_C,
            num_warps=4,
        )

        return x_ln_out


def run(*args):
    return ModelNew()(*args)
