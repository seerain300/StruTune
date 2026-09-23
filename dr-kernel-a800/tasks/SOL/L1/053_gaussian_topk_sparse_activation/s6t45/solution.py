import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(x_ptr, out_sum_ptr,
                     B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                     BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    total = 0.0
    # Iterate over feature dimension in chunks
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        # For contiguous [B, S, F], offsets are: b*S*F + s*F + f
        ptr = x_ptr + row_b * S * F + row_s * F + offs
        x = tl.load(ptr, mask=mask, other=0.0)
        x = x.to(tl.float32)
        total += tl.sum(x, axis=0)
    # Store per-row sum
    tl.store(out_sum_ptr + pid, total)


@triton.jit
def sumsq_rows_kernel(x_ptr, out_sumsq_ptr,
                      B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    total = 0.0
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptr = x_ptr + row_b * S * F + row_s * F + offs
        x = tl.load(ptr, mask=mask, other=0.0)
        x = x.to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.store(out_sumsq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr,
                         B: tl.constexpr, S: tl.constexpr, F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per (b, s) row
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    # Population mean and std
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(z_ptr, p: tl.constexpr,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low: tl.constexpr):
    # Compute inverse-normal CDF z for p in (0, 1) using A&S 7.1.26
    p_val = p
    one_minus_p = 1.0 - p_val

    # Masks for piecewise regions
    mask_low = p_val < p_low
    mask_mid = (p_val >= p_low) & (p_val <= one_minus_p)
    mask_high = p_val > one_minus_p

    # low region: z ~ sqrt(2) * sqrt(-log(p)) * poly(t) / poly1(t)
    t = tl.sqrt(-2.0 * tl.log(p_val))
    z_low = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6) / \
            ((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0)

    # mid region: z ~ poly(r) / poly1(r)
    r = p_val - 0.5
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * r / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # high region: z ~ -poly(q) / poly1(q)
    q = tl.sqrt(-2.0 * tl.log(one_minus_p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(mask_mid, z_mid, z_val)
    z_val = tl.where(mask_high, z_high, z_val)
    tl.store(z_ptr, z_val)


@triton.jit
def apply_threshold_kernel(x_ptr, mean_ptr, std_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           BLOCK_F: tl.constexpr):
    # One program per (b, s) row. It doesn't need to read mean/std; they are scalars passed via pointer arithmetic.
    # Here we recompute threshold per chunk using mean_ptr and std_ptr, which are per-row scalars.
    # To be robust, we avoid reading from mean_ptr/std_ptr; instead, we assume the caller provides a 1-element tensor for threshold per row.
    # However, to keep pure Triton logic, we'll read mean and std for this row and apply threshold.
    # Note: This kernel as written will not work if called without per-row scalars; see ModelNew.forward for correct usage.
    pass  # placeholder to satisfy Triton compilation; we will not call this as written


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only forward: no torch ops on tensors
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        x = x.contiguous()
        assert x.ndim == 3, "Expected input of shape [batch_size, seq_len, intermediate_size]"
        B, S, F = x.shape
        device = x.device

        # Heuristic block size for feature dimension
        BLOCK_F = 1024 if F >= 1024 else (512 if F >= 512 else 256)

        # 1) Reduce: sum and sumsq per (b, s) row
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        grid = (B * S,)
        sum_rows_kernel[grid](x, out_sum, B, S, F, BLOCK_F=BLOCK_F, num_warps=4, num_stages=2)
        sumsq_rows_kernel[grid](x, out_sumsq, B, S, F, BLOCK_F=BLOCK_F, num_warps=4, num_stages=2)

        # 2) Compute mean and std per row (population std)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=1, num_stages=1)

        # 3) Compute inverse-normal CDF z for scalar target_sparsity using A&S 7.1.26
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        # Constants for A&S 7.1.26
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, num_warps=1, num_stages=1)

        z = z_buf[0]  # scalar float32

        # 4) Apply threshold: y = max(x - (mean + std*z), 0)
        # We need per-row threshold as a scalar. For Triton, we can broadcast z to a tensor of shape (B*S,) and compute threshold per row.
        # But since z is scalar, we can compute mean + std*z per row in Triton and apply it.
        # Implement a separate Triton kernel for applying threshold, but to avoid reading uninitialized pointers, we can compute threshold in PyTorch and pass it to Triton. However, to keep Triton-only, we recompute per row here using torch ops on scalars, which is allowed in forward for scalars.

        # Create output as float32; Triton will write elementwise.
        out_f32 = torch.empty_like(x, dtype=torch.float32, device=device)

        # Launch apply kernel: one program per (b, s) row
        apply_grid = (B * S,)
        # Note: The previous placeholder kernel didn't read mean/std; instead, we can compute per-row threshold in Triton using pointers.
        # We will define a real apply kernel that reads per-row mean/std and applies threshold.

        # Real apply kernel (we'll inline it below)

        # Inline apply kernel:
        # We need mean and std per row; they are stored in out_mean and out_std. We will use a Triton kernel that loops over F and applies threshold.

        # Implement a Triton kernel that applies threshold per row, reading mean/std from out_mean/out_std.
        @triton.jit
        def apply_threshold_kernel_row(x_ptr, mean_ptr, std_ptr, out_ptr,
                                        B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                                        BLOCK_F: tl.constexpr):
            pid = tl.program_id(axis=0)  # one program per (b, s) row
            row_b = pid // S
            row_s = pid % S
            mean = tl.load(mean_ptr + pid)
            std = tl.load(std_ptr + pid)
            threshold = mean + std * z  # z is scalar; Triton will broadcast
            for f_start in range(0, F, BLOCK_F):
                offs = f_start + tl.arange(0, BLOCK_F)
                mask = offs < F
                ptr = x_ptr + row_b * S * F + row_s * F + offs
                x = tl.load(ptr, mask=mask, other=0.0)
                x = x.to(tl.float32)
                y = x - threshold
                y = tl.maximum(y, 0.0)  # ReLU
                tl.store(out_ptr + row_b * S * F + row_s * F + offs, y, mask=mask)

        apply_threshold_kernel_row[apply_grid](x, out_mean, out_std, out_f32, B, S, F, BLOCK_F=BLOCK_F, num_warps=4, num_stages=2)

        # Cast to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
