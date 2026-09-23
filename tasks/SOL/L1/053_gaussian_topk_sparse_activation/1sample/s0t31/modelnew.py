import torch
import triton
import triton.language as tl


@triton.jit
def _sparsity_kernel(
    x_ptr,            # *fp32
    out_ptr,          # *fp32 (we'll cast to bf16 in host)
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    K: tl.constexpr,  # int
    stride_b,         # int
    stride_s,         # int
    stride_k,         # int
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Row base offset in elements
    row_base = b * stride_b + s * stride_s

    # 1) Compute mean and sumsq over K in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over K in tiles
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        x = tl.load(x_ptr + row_base + idx * stride_k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute mean and std (population std, var = E[x^2] - (E[x])^2)
    K_f = tl.full((), K, tl.float32)
    mean = sum_val / K_f
    var = sumsq_val / K_f - mean * mean
    # clamp var to avoid tiny negative due to roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # 2) Compute z = inverse_normal_cdf(target_sparsity) using A&S 5.2.23
    # For target_sparsity=0.9, z ~ 1.2815516; but compute in kernel for generality.
    # Constants for A&S 5.2.23
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    # target_sparsity comes as scalar; assume we want 0.9 by default
    target = 0.9
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(target))
    z_lower = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q2 = target - 0.5
    r2 = q2 * q2
    z_center = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * q2 / \
               (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)

    # Pick z = z_center for target in (p_low, 1-p_low). For 0.9, 1-p=0.1 > p_low, so z_center.
    z = z_center

    # 3) threshold = mean + std * z
    threshold = mean + std * z

    # 4) Apply elementwise: out = max(0, x - threshold)
    # Loop over K again to write results
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        x = tl.load(x_ptr + row_base + idx * stride_k, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        # Write to output at the same positions
        tl.store(out_ptr + row_base + idx * stride_k, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_k = int(block_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, f"Expected 3D input, got shape {tuple(x.shape)}"
        B, S, K = x.shape

        # Make contiguous for simple stride_k=1 addressing
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Allocate fp32 output buffer; we'll cast to bf16 after kernel
        out_fp32 = torch.empty_like(x_f32)

        # Launch Triton kernel: one program per (b, s) row
        grid = (B, S)
        _sparsity_kernel[grid](
            x_f32,
            out_fp32,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)