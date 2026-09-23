import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, sum_ptr, sumsq_ptr, B, S, H, NUM_BLOCKS, BLOCK: tl.constexpr):
    # This kernel computes partial sums of x and x^2 for each (b, s) row and atomically adds them into sum_ptr[row] and sumsq_ptr[row].
    # We iterate over the H dimension in chunks of BLOCK, and each program covers a prefix [0, k-1] where k = BLOCK * pid % NUM_BLOCKS.
    # All programs collectively cover [0, H) using NUM_BLOCKS.

    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H

    # Local accumulators (fp32 scalars)
    sum_val = 0.0
    sum_sq = 0.0

    offsets = tl.arange(0, BLOCK)
    # Determine the start index this program should handle
    k = tl.program_id(1) * BLOCK  # pid1 enumerates blocks in [0, NUM_BLOCKS)
    # Loop over prefix with step NUM_BLOCKS
    i = k
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Reduce vector lanes to scalar
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += NUM_BLOCKS

    # Atomically add this program's contribution to global sum and sumsq
    tl.atomic_add(sum_ptr + row, sum_val)
    tl.atomic_add(sumsq_ptr + row, sum_sq)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        BLOCK: tl.constexpr):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF
    # Piecewise logic:
    # if p <= p_low: z = sqrt(-2*log(p))
    # elif p >= p_high: z = sqrt(-2*log(1-p))
    # else: centered polynomial form using p-0.5
    # Store result to out_ptr[0] as fp32
    if p_val <= p_low:
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val >= p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    else:
        q = p_val - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        nd = poly * q / den

    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0] as fp32
    z = tl.load(z_ptr)

    # Compute threshold (fp32)
    thr = mean + std * z

    # Base offset for this (b, s) row
    base = b * S * H + s * H

    # Apply y = max(x - thr, 0) elementwise, write as bfloat16
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)  # load as fp32
        diff = x - thr
        y = tl.maximum(diff, 0.0)  # ReLU
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # input_tensor: [batch_size, seq_len, intermediate_size]
        # target_sparsity: float in [0, 1], 0 means no sparsity

        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return input_tensor.to(torch.bfloat16)

        B, S, H = input_tensor.shape
        device = input_tensor.device

        # 1) Compute per-row sum and sumsq in fp32 using Triton with atomic accumulation
        # Create sum and sumsq buffers
        sum_buf = torch.zeros((B * S,), dtype=torch.float32, device=device)
        sumsq_buf = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Ensure input is contiguous and fp32
        in_fp32 = input_tensor.contiguous().to(torch.float32)

        # Number of blocks along H for reduction
        BLOCK = 1024
        NUM_BLOCKS = (H + BLOCK - 1) // BLOCK  # total number of chunks per row
        grid_stats = (B * S, NUM_BLOCKS)

        # Launch Triton kernel to accumulate sums
        compute_mean_std_fp32[grid_stats](in_fp32, sum_buf, sumsq_buf, B, S, H, NUM_BLOCKS, BLOCK=BLOCK)

        # 2) Compute mean and std per row
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=device)

        # mean = sum / H, std = sqrt(sumsq / H - mean^2)  (population, unbiased=False)
        mean_buf = sum_buf / float(H)
        # Using Triton kernel to perform std computation elementwise (micro-op)
        # Here we just do elementwise ops on tensor: Triton kernels must be used for math.
        # To satisfy Triton-only, we implement a tiny kernel that computes std for each row.
        @triton.jit
        def compute_std_from_sums(mean_ptr, sumsq_ptr, std_ptr, N: tl.constexpr):
            row = tl.program_id(0)
            m = tl.load(mean_ptr + row)
            ss = tl.load(sumsq_ptr + row)
            n = N  # N=H
            var = ss / n - m * m
            std = tl.sqrt(var)
            tl.store(std_ptr + row, std)

        compute_std_from_sums[(B * S,)](mean_buf, sumsq_buf, std_buf, H)

        # 3) Compute icdf(target_sparsity) as fp32 scalar on device using Triton (A&S approximation)
        p_val = float(target_sparsity)
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

        z_buf = torch.empty((), dtype=torch.float32, device=device)
        compute_icdf_scalar[(1,)](z_buf, p_val, p_low, p_high,
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4,
                                  BLOCK=128)

        # 4) Apply thresholding: y = max(x - (mean + std*z), 0) in fp32, then cast to bfloat16
        out_fp32 = torch.empty_like(in_fp32)  # fp32 buffer for output (we cast later)
        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](in_fp32, out_fp32, mean_buf, std_buf, B, S, H, z_buf, BLOCK=1024)

        # 5) Cast output to bfloat16 to match original code's return dtype
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
