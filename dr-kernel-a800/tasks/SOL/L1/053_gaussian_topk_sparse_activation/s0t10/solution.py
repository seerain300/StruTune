import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean across feature dimension (F) for a 2D tensor [NROWS, F]
@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,   # *fp32, shape [NROWS, F], contiguous
    mean_out_ptr, # *fp32, shape [NROWS]
    F,            # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_sum = 0.0
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        idx = pid * F + offs
        x = tl.load(inputs_ptr + idx, mask=mask, other=0.0)
        row_sum += tl.sum(x, axis=0)
    mean = row_sum / F
    tl.store(mean_out_ptr + pid, mean)


# Kernel: compute per-row std across feature dimension (F) for a 2D tensor [NROWS, F]
# population std: std = sqrt(sum((x - mean)^2) / F)
@triton.jit
def std_lastdim_kernel(
    inputs_ptr,   # *fp32, shape [NROWS, F], contiguous
    mean_ptr,     # *fp32, shape [NROWS]
    std_out_ptr,  # *fp32, shape [NROWS]
    F,            # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid)
    sum_sq = 0.0
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        idx = pid * F + offs
        x = tl.load(inputs_ptr + idx, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    std = tl.sqrt(sum_sq / F)
    tl.store(std_out_ptr + pid, std)


# Kernel: compute inverse-normal CDF (ndtri) for scalar p using A&S 5.2.23
# Writes the scalar to z_ptr[0]
@triton.jit
def ndtri_approx_kernel(
    z_ptr,        # *fp32, shape [1]
    p,            # fp32 scalar, target_sparsity
    p_low,        # fp32
    p_high,       # fp32
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
):
    # Compute ndtri for scalar p via A&S 5.2.23
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / denom
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = -poly / denom
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        z = (poly * q) / denom
    tl.store(z_ptr, z)


# Kernel: apply cutoff = mean + std * z and do y = max(0, x - cutoff) elementwise
@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,    # *fp32, [NROWS, F], contiguous
    mean_ptr,      # *fp32, [NROWS]
    std_ptr,       # *fp32, [NROWS]
    z_ptr,         # *fp32, [1], scalar z
    out_ptr,       # *fp32, [NROWS, F], contiguous
    F,             # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar z
    cutoff = mean + std * z

    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        idx = pid * F + offs
        x = tl.load(inputs_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return original (no-op)
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and dtype for compute
        assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Compute in fp32 for numeric stability
        inputs_f32 = inputs.to(torch.float32)

        # Reshape to 2D [NROWS, F] and make contiguous. NROWS = batch_size * seq_len
        B, S, F = inputs_f32.shape
        NROWS = B * S
        inputs_2d = inputs_f32.view(NROWS, F).contiguous()

        # Allocate outputs (fp32 for compute)
        mean_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)
        std_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)
        out_fp32 = torch.empty((NROWS, F), dtype=torch.float32, device=inputs.device)

        # Launch mean kernel: one program per row
        mean_lastdim_kernel[(NROWS,)](
            inputs_2d, mean_out, F, BLOCK_F=2048, num_warps=4
        )

        # Launch std kernel: one program per row
        std_lastdim_kernel[(NROWS,)](
            inputs_2d, mean_out, std_out, F, BLOCK_F=2048, num_warps=4
        )

        # Compute ndtri(z) for scalar target_sparsity (inverse-normal CDF) via A&S 5.2.23
        p_low = 0.02425
        p_high = 1.0 - p_low
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

        # 1-element buffer for z on device
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Launch ndtri kernel (single program)
        ndtri_approx_kernel[(1,)](
            z_buf, float(target_sparsity), p_low, p_high, a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4,
            num_warps=1,
        )

        # Apply cutoff and ReLU: one program per row
        apply_cutoff_relu_kernel[(NROWS,)](
            inputs_2d, mean_out, std_out, z_buf, out_fp32, F, BLOCK_F=2048, num_warps=4
        )

        # Reshape back to [B, S, F] and cast to original dtype (bfloat16 in tests)
        out_3d = out_fp32.view(B, S, F)
        return out_3d.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
