import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const bfloat16, input [num_rows, features]
    y_ptr,            # *bfloat16, output [num_rows, features]
    ln_weight_ptr,    # *const float32, [features]
    ln_bias_ptr,      # *const float32, [features]
    num_rows,         # int32
    features,         # int32
    eps,              # float32
    BLOCK: tl.constexpr,  # tile size for reduction (e.g., 1024)
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    # First pass: compute sum and sum of squares in fp32
    sum_fp32 = 0.0
    sumsq_fp32 = 0.0

    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row * features + idx, mask=mask, other=0.0).to(tl.float32)
        sum_fp32 += tl.sum(x, axis=0)
        sumsq_fp32 += tl.sum(x * x, axis=0)

    mean = sum_fp32 / features
    var = sumsq_fp32 / features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row * features + idx, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        out = norm * w + b  # fp32
        tl.store(y_ptr + row * features + idx, out.to(tl.bfloat16), mask=mask)


@triton.jit
def gelu_kernel_fp32(
    x_ptr,    # *const float32, input flattened [N]
    y_ptr,    # *float32, output [N]
    N,        # int32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # GELU via erf approximation (Abramowitz & Stegun 7.1.26)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    t = x * inv_sqrt2
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    sign = tl.where(t >= 0, 1.0, -1.0)
    abs_t = tl.abs(t)
    # Horner's method for polynomial
    t2 = t * t
    poly = a5 * t2 + a4
    poly = poly * t2 + a3
    poly = poly * t2 + a2
    poly = poly * t2 + a1
    poly = poly * t2  # t^10
    poly = poly * t2  # t^12
    poly = poly * t2  # t^14 -> t^16, but we only need up to t^5
    # Fix: compute correctly up to t^5
    poly = a1 * t + a2 * (t * t) + a3 * (t * t * t) + a4 * (t * t * t * t) + a5 * (t * t * t * t * t)
    erf_t = sign * (1.0 - poly * tl.exp(-t * t))
    y = 0.5 * x * (1.0 + erf_t)
    tl.store(y_ptr + offs, y, mask=mask)


def triton_layernorm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    # hidden: [num_patches, 1536], bfloat16
    num_rows = hidden.shape[0]
    features = hidden.shape[1]
    x = hidden.contiguous()
    y = torch.empty_like(x, dtype=torch.bfloat16)
    lw = ln_weight.to(torch.float32).contiguous()
    lb = ln_bias.to(torch.float32).contiguous()
    BLOCK = 1024
    grid = (num_rows,)
    layernorm_row_kernel[grid](x, y, lw, lb, num_rows, features, eps, BLOCK=BLOCK, num_warps=4)
    return y


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    # x: fp32 tensor, flatten for elementwise kernel
    N = x.numel()
    y = torch.empty_like(x, dtype=torch.float32)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    # Flatten for kernel, then reshape back
    x_flat = x.reshape(-1)
    gelu_kernel_fp32[grid](x_flat, y.reshape(-1), N, BLOCK=BLOCK, num_warps=4)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # 1) LayerNorm in Triton (per-row on 1536 features)
        hidden_norm = triton_layernorm(hidden, ln_weight, ln_bias, eps)

        # 2) Spatial shuffle via PyTorch permute/view exactly as original
        # Replicate the original logic for each grid: t, h, w, compute num_patches_this,
        # then permute and reshape.
        num_grids = grid_thw.shape[0]
        offset = 0
        merged = []
        for i in range(num_grids):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches = hidden_norm[offset:offset + num_patches_this]
            h_merged = h // 2
            w_merged = w // 2
            patches = patches.view(t, h_merged, 2, w_merged, 2, patches.shape[-1])
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(t * h_merged * w_merged, 32 * patches.shape[-1])
            merged.append(patches)
            offset += num_patches_this
        hidden_shuffled = torch.cat(merged, dim=0)  # float32 from original hidden_norm

        # 3) First Linear in PyTorch (matmul + bias), keep fp32 compute
        A = hidden_shuffled  # [num_merged_patches, 12288] fp32
        B = fc1_weight.t().to(torch.float32).contiguous()  # [12288, 6144]
        out1 = A.matmul(B) + fc1_bias.to(torch.float32)    # [num_merged_patches, 6144] fp32

        # 4) GELU in Triton (fp32), result remains fp32
        out1_gelu = triton_gelu(out1)

        # 5) Second Linear in PyTorch
        B2 = fc2_weight.t().to(torch.float32).contiguous()  # [6144, 3584]
        out2 = out1_gelu.matmul(B2) + fc2_bias.to(torch.float32)  # fp32

        return out2


def run(*args):
    return ModelNew()(*args)
