import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all computation inside Triton
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_rows_kernel(
        x_ptr,            # *const float32, flattened view of [B, S, F]
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B,                # int
        S,                # int
        F,                # int
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (b, s)
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        out_idx = b * S + s
        tl.store(sums_ptr + out_idx, local_sum)
        tl.store(sums2_ptr + out_idx, local_sumsq)

    @triton.jit
    def compute_mean_std_rows_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F,                # int
    ):
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        out_idx = b * S + s

        total = tl.load(sums_ptr + out_idx)
        total2 = tl.load(sums2_ptr + out_idx)

        mean = total / F
        var = total2 / F - mean * mean
        var = tl.maximum(var, 0.0)  # avoid tiny negative due to FP
        std = tl.sqrt(var)

        tl.store(mean_ptr + out_idx, mean)
        tl.store(std_ptr + out_idx, std)

    @triton.jit
    def ndtri_kernel(
        out_ptr,          # *float32, length 1
        p,                # float32 scalar target_sparsity
    ):
        # Abramowitz and Stegun 7.1.26 approximation for inverse normal CDF
        p = tl.maximum(tl.minimum(p, 1.0), 0.0)  # clamp to [0, 1]

        # Constants
        p1 = 0.319381530
        p2 = -0.356563782
        p3 = 1.781477937
        p4 = -1.821255978
        p5 = 1.330274429

        # Compute w = sqrt( -2 * (log(p) - log(1-p)) )
        log_p = tl.log(p)
        log_1m = tl.log(1.0 - p)
        w = tl.sqrt(-2.0 * (log_p - log_1m))

        # Polynomial in t = 1 / (1 + p1*w)
        t = 1.0 / (1.0 + p1 * w)
        poly = (((((p5 * t + p4) * t + p3) * t + p2) * t + p1) * t)

        z = (1.0 / w) * poly
        # sign: for p in (0.5, 1), z is positive; for p in (0, 0.5), z is negative
        sign = tl.where(p > 0.5, 1.0, -1.0)
        z = sign * z

        tl.store(out_ptr, z)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32
        out_ptr,          # *float32
        mean_ptr,         # *const float32, length B*S
        std_ptr,          # *const float32, length B*S
        ndtri_scale,      # float32 scalar
        B,                # int
        S,                # int
        F,                # int
        total_elems,      # int
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

        out_idx = b * S + s
        mean = tl.load(mean_ptr + out_idx, mask=mask, other=0.0)
        std = tl.load(std_ptr + out_idx, mask=mask, other=0.0)

        threshold = mean + std * ndtri_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU

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

        # Ensure float32 and contiguous
        x = inputs.to(torch.float32).contiguous()
        assert x.ndim == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, F = x.shape
        device = x.device

        # Flatten for reduction; Triton expects a flat pointer
        x_flat = x.reshape(B * S * F)

        # 1) Reduce per (b, s) row: sums and sums2
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_reduce = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x_flat, sums, sums2, B, S, F, BLOCK_SIZE=1024, num_warps=4
        )

        # 2) Compute mean and std per row
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_stats = (B * S,)
        compute_mean_std_rows_kernel[grid_stats](
            sums, sums2, mean, std, F, num_warps=1
        )

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity (scalar)
        ndtri_scale = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_kernel[(1,)](ndtri_scale, float(target_sparsity), num_warps=1)

        # 4) Apply sparsification: y = max(0, x - (mean + std * ndtri_scale))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)
        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        grid_sp = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_sp](
            x_flat, out_fp32, mean, std, ndtri_scale[0], B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT, num_warps=4
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
