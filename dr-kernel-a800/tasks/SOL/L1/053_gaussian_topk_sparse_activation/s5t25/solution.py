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
        x_ptr,            # *const float32, input flattened
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size (constexpr for unrolling)
        S: tl.constexpr,  # seq_len (constexpr for unrolling)
        F,                # feature_size (intermediate_size), runtime int
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        row = pid
        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row * F + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + row, local_sum)
        tl.store(sums2_ptr + row, local_sumsq)


    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F: tl.constexpr,  # feature_size (runtime int but used as scalar)
    ):
        # One program per element (B*S)
        pid = tl.program_id(axis=0)
        s = tl.load(sums_ptr + pid)
        s2 = tl.load(sums2_ptr + pid)
        mean = s / float(F)
        var = s2 / float(F) - mean * mean
        # Clamp variance to non-negative to avoid tiny negative due to FP error
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)


    @triton.jit
    def ndtri_scalar_kernel(
        p,                 # scalar float32, target sparsity (0,1)
        out_ptr,           # *float32, single element
        P_LOW: tl.constexpr = 0.02425,
        BLOCK_SIZE: tl.constexpr = 1,
    ):
        # Abramowitz and Stegun 7.1.26 approximation for ndtri(p)
        # Single scalar compute; branch lower/upper regions.
        if p <= P_LOW:
            q = tl.sqrt(-2.0 * tl.log(p))
            a1 = tl.full((), -3.969683028665376e+01, tl.float32)
            a2 = tl.full((),  2.209460984245205e+02, tl.float32)
            a3 = tl.full((), -2.759285104469687e+02, tl.float32)
            a4 = tl.full((),  1.383577518672690e+02, tl.float32)
            a5 = tl.full((), -3.066479806614716e+01, tl.float32)
            a6 = tl.full((),  2.506628277459239e+00, tl.float32)

            b1 = tl.full((), -5.447609879822406e+01, tl.float32)
            b2 = tl.full((),  1.615858368580409e+02, tl.float32)
            b3 = tl.full((), -1.556989798598866e+02, tl.float32)
            b4 = tl.full((),  6.680131188771972e+01, tl.float32)
            b5 = tl.full((), -1.328068155288572e+01, tl.float32)

            c1 = tl.full((), -7.784894002430293e-03, tl.float32)
            c2 = tl.full((), -3.223964580411365e-01, tl.float32)
            c3 = tl.full((), -2.400758277161838e+00, tl.float32)
            c4 = tl.full((), -2.549732539343734e+00, tl.float32)
            c5 = tl.full((),  4.374664141464968e+00, tl.float32)
            c6 = tl.full((),  2.938163982698783e+00, tl.float32)

            d1 = tl.full((),  7.784695709041462e-03, tl.float32)
            d2 = tl.full((),  3.224671290700398e-01, tl.float32)
            d3 = tl.full((),  2.445134137142996e+00, tl.float32)
            d4 = tl.full((),  3.754408661907416e+00, tl.float32)

            poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
            qpoly = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
            result = poly / qpoly
            tl.store(out_ptr, result)
        else:
            qm = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (((((c1*qm + c2)*qm + c3)*qm + c4)*qm + c5)*qm + c6)
            qpoly = (((((d1*qm + d2)*qm + d3)*qm + d4)*qm + 1.0))
            result = - (poly / qpoly)
            tl.store(out_ptr, result)


    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        ndtri_scalar_ptr, # *const float32, single element
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

        # Load input
        x_ptrs = x_ptr + b * SF + s * F + f
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load mean and std for this (b, s)
        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        # Load ndtri scalar
        scale = tl.load(ndtri_scalar_ptr)  # scalar

        threshold = mean + std * scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
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
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_REDUCE
        )

        # 2) Compute mean and std in Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)
        compute_mean_std_kernel[(B * S,)](
            sums, sums2, mean, std, F=F
        )

        # 3) Compute ndtri(target_sparsity) using Triton kernel
        ndtri_scalar = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_scalar_kernel[(1,)](target_sparsity, ndtri_scalar)

        # 4) Apply sparsification: output = max(0, x - threshold) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 4096
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, ndtri_scalar, B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
