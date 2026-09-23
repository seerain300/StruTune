import torch
import math

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
    def reduce_row_sum_sumsq_kernel(
        x_ptr,            # *const float32, flattened input
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        F,                # feature_size (intermediate dimension)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        row_start = pid * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over the feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        # Store per-row results (one program per row, no atomics needed)
        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)


    @triton.jit
    def inv_erf_kernel(
        out_ptr,              # *float32, scalar output
        p,                    # float32, input probability in (0,1)
        A1, A2, A3, A4, A5,  # float32 constants for A
        B1, B2, B3, B4, B5,  # float32 constants for B
        C1, C2, C3, C4, C5, C6,  # float32 constants for C
        D1, D2, D3, D4,      # float32 constants for D
    ):
        # Abramowitz and Stegun 7.1.26: z = sqrt(2)*erfinv(2p - 1) via rational approximations.
        t = 2.0 * p - 1.0
        # Polynomial A
        a1 = A1; a2 = A2; a3 = A3; a4 = A4; a5 = A5
        poly_mid = (((((a1 * t) + a2) * t + a3) * t + a4) * t + a5) * t + 1.0
        poly_mid = poly_mid * t
        # Polynomial B
        b1 = B1; b2 = B2; b3 = B3; b4 = B4; b5 = B5
        denom_mid = (((((b1 * t) + b2) * t + b3) * t + b4) * t + b5) * t + 1.0
        z_mid = poly_mid / denom_mid
        tl.store(out_ptr, z_mid)


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
        y = tl.maximum(y, 0.0)  # ReLU

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure Triton availability and CUDA tensors
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Convert to float32 for statistics; ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        total_elems = B * S * F
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_SIZE_REDUCE = 1024
        grid_reduce = (B * S,)
        reduce_row_sum_sumsq_kernel[grid_reduce](
            x, sums, sums2, F, BLOCK_SIZE=BLOCK_SIZE_REDUCE
        )

        # 2) Compute mean and std per (b, s) on host
        mean = sums / float(F)
        var = sums2 / float(F) - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # 3) Compute threshold scale (ndtri(target_sparsity)) using Triton kernel
        # Abramowitz and Stegun 7.1.26 approximation for erfinv; z = sqrt(2)*erfinv(2p-1)
        A1 = -1.381551055796698
        A2 = 0.361127384403674
        A3 = -0.094482498297331
        A4 = 0.078137326538672
        A5 = -0.031223373600928
        B1 = -4.433816845333261
        B2 = 1.935916034185939
        B3 = -0.603118321792633
        B4 = 0.287162667281933
        B5 = -0.138938950259879
        C1 = -7.784894002430293e-03
        C2 = -3.223964580411365e-01
        C3 = -2.400758277161838e+00
        C4 = -2.549732539343734e+00
        C5 = 4.374664141464968e+00
        C6 = 2.938163982698783e+00
        D1 = 7.784695709041462e-03
        D2 = 3.224671290700398e-01
        D3 = 2.445134137142996e+00
        D4 = 3.754408661907416e+00

        threshold_scale_buf = torch.empty(1, dtype=torch.float32, device=device)
        inv_erf_kernel[(1,)](
            threshold_scale_buf,
            torch.tensor(target_sparsity, dtype=torch.float32, device=device),
            A1, A2, A3, A4, A5,
            B1, B2, B3, B4, B5,
            C1, C2, C3, C4, C5, C6,
            D1, D2, D3, D4,
        )
        threshold_scale = threshold_scale_buf[0] * math.sqrt(2.0)

        # 4) Apply sparsification: output = max(0, x - threshold) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, threshold_scale,
            B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
