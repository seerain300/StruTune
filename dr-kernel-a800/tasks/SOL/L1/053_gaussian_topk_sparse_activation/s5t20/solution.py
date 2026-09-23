import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Triton kernels
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_rows_kernel(
        x_ptr,            # *const float32, input laid out as [B*S, F]
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        F,                # feature_size
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row
        row_id = tl.program_id(axis=0)
        base = row_id * F
        acc_sum = 0.0
        acc_sumsq = 0.0
        # Loop over feature dimension
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + base + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            acc_sum += tl.sum(vals, axis=0)
            acc_sumsq += tl.sum(vals * vals, axis=0)
        tl.store(sums_ptr + row_id, acc_sum)
        tl.store(sums2_ptr + row_id, acc_sumsq)

    @triton.jit
    def compute_mean_std_rows_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F: tl.constexpr,
    ):
        row_id = tl.program_id(axis=0)
        sum_row = tl.load(sums_ptr + row_id)
        sumsq_row = tl.load(sums2_ptr + row_id)
        mean = sum_row / F
        var = sumsq_row / F - mean * mean
        # Clamp to avoid tiny negatives due to FP rounding
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + row_id, mean)
        tl.store(std_ptr + row_id, std)

    @triton.jit
    def inv_normal_cdf_scalar_kernel(
        out_ptr,          # *float32, single-element output
        p,                # float32 scalar in (0,1)
        # coefficients for Abramowitz and Stegun 7.1.26 approximation
        a1 = -3.969683028665376e+01,
        a2 = 2.209460984245205e+02,
        a3 = -2.759285104469687e+02,
        a4 = 1.383577518672690e+02,
        a5 = -3.066479806614716e+01,
        a6 = 2.506628277459239e+00,
        b1 = -5.447609879822406e+01,
        b2 = 1.615858368580409e+02,
        b3 = -1.556989798598866e+02,
        b4 = 6.680131188771972e+01,
        b5 = -1.328068155288572e+01,
        c1 = -7.784894002430293e-03,
        c2 = -3.223964580411365e-01,
        c3 = -2.400758277161838e+00,
        c4 = -2.549732539343734e+00,
        c5 = 4.374664141464968e+00,
        c6 = 2.938163982698783e+00,
        d1 = 7.784695709041462e-03,
        d2 = 3.224671290700398e-01,
        d3 = 2.445134137142996e+00,
        d4 = 3.754408661907416e+00,
        p_low = 0.02425,
    ):
        # Compute inverse normal CDF for p using A&S approximation
        if p < p_low:
            q = tl.sqrt(-2.0 * tl.log(p))
            poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            dpoly = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
            y = poly / dpoly
        elif p > (1.0 - p_low):
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            dpoly = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
            y = -poly / dpoly
        else:
            q = p - 0.5
            r = q * q
            poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            dpoly = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            y = poly / dpoly
        tl.store(out_ptr, y)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,      # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elems

        SF = S * F
        b = offs // SF
        rem = offs % SF
        s = rem // F
        f = rem % F

        x_ptrs = x_ptr + b * SF + s * F + f
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        threshold = mean + std * threshold_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If Triton not available or input not on CUDA, fallback to original PyTorch computation
        if not TRITON_AVAILABLE or not inputs.is_cuda:
            if target_sparsity == 0.0:
                return inputs
            inputs_f32 = inputs.to(torch.float32)
            inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
            inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
            # Original used erfinv; match behavior
            std_multiplier = torch.special.erfinv(2.0 * torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device) - 1.0)
            cutoff_threshold = inputs_mean + inputs_std * std_multiplier
            sparse_output = F.relu(inputs_f32 - cutoff_threshold)
            return sparse_output.to(torch.bfloat16)

        # Triton path
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        rows = B * S
        sums = torch.empty(rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(rows, dtype=torch.float32, device=device)

        grid_reduce = (rows,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](x, sums, sums2, F, BLOCK_SIZE=1024)

        # 2) Compute mean and std per row in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=device)
        std = torch.empty(rows, dtype=torch.float32, device=device)

        grid_meanstd = (rows,)
        compute_mean_std_rows_kernel[grid_meanstd](sums, sums2, mean, std, F)

        # 3) Compute inverse normal CDF for target_sparsity in Triton (scalar)
        threshold_scale = torch.empty(1, dtype=torch.float32, device=device)
        p = float(target_sparsity)  # pass as Python float to kernel; inside Triton we use it as scalar
        grid_ndtri = (1,)
        inv_normal_cdf_scalar_kernel[grid_ndtri](threshold_scale, p)

        threshold_scale_val = float(threshold_scale.item())

        # 4) Apply sparsity: out = max(0, x - (mean + std * threshold_scale)) in Triton
        out_fp32 = torch.empty_like(x, dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE = 1024
        grid_sparsify = (triton.cdiv(total_elems, BLOCK_SIZE),)
        sparsify_relu_kernel[grid_sparsify](
            x, out_fp32, mean, std, threshold_scale_val, B, S, F, total_elems, BLOCK_SIZE
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
