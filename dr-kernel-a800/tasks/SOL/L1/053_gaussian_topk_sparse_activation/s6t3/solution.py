import torch
import triton
import triton.language as tl


# Kernel: compute per-(b, s) row sums across feature dimension F
@triton.jit
def sum_rows_kernel(input_ptr, out_sum_ptr,
                    B, S, F, stride_b, stride_s, stride_f,
                    BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = input_ptr + base + idx * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
    tl.atomic_add(out_sum_ptr + pid, sum_val)


# Kernel: compute per-(b, s) row sum of squares across feature dimension F
@triton.jit
def sumsq_rows_kernel(input_ptr, out_sumsq_ptr,
                      B, S, F, stride_b, stride_s, stride_f,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    sumsq_val = 0.0
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = input_ptr + base + idx * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        sumsq_val += tl.sum(vals * vals, axis=0)
    tl.atomic_add(out_sumsq_ptr + pid, sumsq_val)


# Kernel: compute per-(b, s) mean and std from sum and sumsq
# std is population std = sqrt(sumsq/F - mean^2)
@triton.jit
def compute_stats_kernel(sum_ptr, sumsq_ptr, mean_out_ptr, std_out_ptr,
                         B, S, F):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    sum_val = tl.load(sum_ptr + pid)
    sumsq_val = tl.load(sumsq_ptr + pid)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_out_ptr + pid, mean)
    tl.store(std_out_ptr + pid, std)


# Kernel: compute inverse-normal CDF (quantile) for a single scalar p
# Implements Abramowitz & Stegun 7.1.26 approximation (A&S formula 26.2.23).
@triton.jit
def ndtri_scalar_kernel(z_out_ptr,
                        p,  # scalar float, passed from host
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low):
    # Create a 1-element buffer for p in float32
    p_buf = tl.full((1,), p, tl.float32)
    # Piecewise computation without Python ifs
    # Lower region: p < p_low
    mask_low = p_buf < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p_buf))  # for p < p_low, q = sqrt(-2*log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # Central region: p_low <= p <= (1 - p_low)
    mask_mid = (p_buf >= p_low) & (p_buf <= (1.0 - p_low))
    p_mid = p_buf
    q_mid = p_mid - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / den_mid
    # Upper region: p > (1 - p_low)
    mask_up = p_buf > (1.0 - p_low)
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p_buf))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    # Select via masks; since p is a single scalar, one mask will be true
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_up, z_up, 0.0)
    # Store to z_out_ptr[0]
    tl.store(z_out_ptr, z)


# Kernel: elementwise apply threshold and ReLU
# For each (b, s) row, load mean and std, compute threshold = mean + std * z,
# then compute y = max(input - threshold, 0). Store bfloat16 output.
@triton.jit
def apply_threshold_kernel(input_ptr, mean_ptr, std_ptr, z_ptr, output_ptr,
                           B, S, F, stride_b, stride_s, stride_f,
                           BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # single scalar

    threshold = mean + std * z

    # Process the row in chunks of BLOCK_F
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        in_ptrs = input_ptr + base + idx * stride_f
        in_vals = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = in_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        # store as bfloat16
        out_ptrs = output_ptr + base + idx * stride_f
        tl.store(out_ptrs, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.0):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: [B, S, F]
        assert inputs.dim() == 3, "Input must be 3D [batch_size, seq_len, intermediate_size]"
        B, S, F = inputs.shape
        device = inputs.device

        # Prepare output
        output = torch.empty((B, S, F), dtype=torch.bfloat16, device=device)

        # Ensure input is contiguous for simple striding
        inp = inputs.contiguous()

        # Compute strides
        stride_b = inp.stride(0)
        stride_s = inp.stride(1)
        stride_f = inp.stride(2)

        # Allocate per-row sums and sumsq
        sum_buf = torch.zeros((B * S,), dtype=torch.float32, device=device)
        sumsq_buf = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels
        BLOCK_F = 256  # chunk size for feature dimension; tuneable
        grid = (B * S,)
        sum_rows_kernel[grid](
            inp,
            sum_buf,
            B, S, F, stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
        )
        sumsq_rows_kernel[grid](
            inp,
            sumsq_buf,
            B, S, F, stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
        )

        # Allocate per-row mean and std
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Compute mean and std (population std, unbiased=False)
        compute_stats_kernel[grid](
            sum_buf, sumsq_buf, mean_buf, std_buf,
            B, S, F,
        )

        # Compute inverse-normal CDF for target_sparsity (A&S 7.1.26)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)  # 1-element buffer

        # Constants (same as original _ndtri function)
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

        ndtri_scalar_kernel[(1,)](
            z_buf,
            self.target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
        )

        # Apply thresholding and ReLU
        apply_threshold_kernel[grid](
            inp,
            mean_buf,
            std_buf,
            z_buf,
            output,
            B, S, F, stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
        )

        return output


def run(*args):
    return ModelNew()(*args)
