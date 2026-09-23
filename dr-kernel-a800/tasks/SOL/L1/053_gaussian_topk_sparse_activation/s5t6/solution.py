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
        x_ptr,            # *const float32
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size (not strictly needed inside, but kept for clarity)
        S: tl.constexpr,  # seq_len (not strictly needed inside, but kept for clarity)
        F,                # feature_size (intermediate dimension)
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in tiles of 256
        for offs in range(0, F, 256):
            idx = offs + tl.arange(0, 256)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F,                # feature_size
    ):
        pid = tl.program_id(axis=0)
        sumv = tl.load(sums_ptr + pid)
        sumsqv = tl.load(sums2_ptr + pid)

        mean = sumv / F
        var = sumsqv / F - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_scalar_kernel(
        out_ptr,          # *float32, scalar output
        p,                # float32 input in (0, 1)
    ):
        # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF.
        # We implement the erf-based approach:
        # erf(z) ≈ 1 - 2 / sqrt(pi) * exp(-z^2), z > 0
        # Solve for z: erfinv(x) ≈ sqrt(pi/2) * z, where erf(z) = x.
        # Constants for approximation:
        a1 = -3.9696830e+01
        a2 = 2.2094609e+02
        a3 = -2.7592851e+02
        a4 = 1.3835775e+02
        a5 = -3.0664799e+01
        a6 = 2.5066283e+00
        b1 = -5.4476099e+01
        b2 = 1.6158584e+02
        b3 = -1.5569898e+02
        b4 = 6.6801312e+01
        b5 = -1.3280682e+01
        c1 = -7.7848940e-03
        c2 = -3.2239646e-01
        c3 = -2.4007583e+00
        c4 = -2.5497325e+00
        c5 = 4.3746641e+00
        c6 = 2.9381640e+00
        d1 = 7.7846957e-03
        d2 = 3.2246713e-01
        d3 = 2.4451341e+00
        d4 = 3.7544087e+00

        p = tl.cast(p, tl.float32)
        if p <= 0.5:
            # p <= 0.5 branch
            t = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = inv  # positive since p <= 0.5
        else:
            # p > 0.5 branch
            t = tl.sqrt(-2.0 * tl.log(p))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = -inv  # negative since p > 0.5

        tl.store(out_ptr, result)


    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,      # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # 1D grid over total elements
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
        # ReLU
        y = tl.maximum(y, 0.0)

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only execution
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure contiguous and compute in float32
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        total_rows = B * S

        # 1) Reduce per row: sum and sum of squares (one program per row)
        sums = torch.empty(total_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(total_rows, dtype=torch.float32, device=device)
        reduce_sum_sumsq_rows_kernel[(total_rows,)](x, sums, sums2, B, S, F)

        # 2) Compute mean and std per row
        mean = torch.empty(total_rows, dtype=torch.float32, device=device)
        std = torch.empty(total_rows, dtype=torch.float32, device=device)
        compute_mean_std_kernel[(total_rows,)](sums, sums2, mean, std, F)

        # 3) Compute inverse normal CDF (ndtri) multiplier in Triton (scalar)
        # Only if target_sparsity in (0, 1); otherwise default to 0.
        if not (0.0 < target_sparsity < 1.0):
            threshold_scale = 0.0
        else:
            scale_buf = torch.empty(1, dtype=torch.float32, device=device)
            ndtri_scalar_kernel[(1,)](scale_buf, target_sparsity)
            threshold_scale = scale_buf[0].item()  # read back as Python float

        # 4) Sparsify: output = max(0, x - (mean + std * threshold_scale))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elems, BLOCK_SIZE),)
        sparsify_relu_kernel[grid](
            x, out_fp32, mean, std, threshold_scale, B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
