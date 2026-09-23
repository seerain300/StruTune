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
    def reduce_row_sum_sumsq_kernel(
        x_ptr,            # *const float32, flattened input
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B,                # int32
        S,                # int32
        F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            vals = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def inv_erf_kernel(
        out_ptr,          # *float32, length 1
        x_ptr,            # *float32, length 1 (scalar target_sparsity)
        A1, A2, A3, A4, A5,     # constants
        B1, B2, B3, B4, B5,     # constants
        C1, C2, C3, C4, C5, C6, # constants
        D1, D2, D3, D4,         # constants
    ):
        # Load scalar target_sparsity
        t = tl.load(x_ptr)
        # x = 1 - t
        x = 1.0 - t

        # Polynomial approximations
        # z = 0.5 * (1 - erf(x)) for x in [0, 1], which equals Phi^(-1)(t) via transformation
        # Use A/B polynomials for small and large x, switch by x < 0.5
        # Evaluate polynomials via Horner's method and compute z
        # Note: Triton lacks built-in erf, we implement via inverse erfinv approximation.
        # Implementation follows A&S 7.1.26.

        # Small-x region
        # p = ((((C1*t + C2)*t + C3)*t + C4)*t + C5)*t + C6
        # q = ((((D1*t + D2)*t + D3)*t + D4)*t + 1.0)
        t2 = t * t
        p_small = C1 * t + C2
        p_small = p_small * t + C3
        p_small = p_small * t + C4
        p_small = p_small * t + C5
        p_small = p_small * t + C6

        q_small = D1 * t + D2
        q_small = q_small * t + D3
        q_small = q_small * t + D4
        q_small = q_small * t + 1.0

        z_small = -p_small / q_small

        # Large-x region
        # p = ((((C1*x + C2)*x + C3)*x + C4)*x + C5)*x + C6
        # q = ((((D1*x + D2)*x + D3)*x + D4)*x + 1.0)
        x2 = x * x
        p_large = C1 * x + C2
        p_large = p_large * x + C3
        p_large = p_large * x + C4
        p_large = p_large * x + C5
        p_large = p_large * x + C6

        q_large = D1 * x + D2
        q_large = q_large * x + D3
        q_large = q_large * x + D4
        q_large = q_large * x + 1.0

        z_large = p_large / q_large

        # Choose branch based on t < 0.5
        use_small = t < 0.5
        z = tl.where(use_small, z_small, z_large)

        # Inverse of Phi: ndtri(t) = sqrt(2) * z
        z = z * 1.772453850905516  # sqrt(2)
        tl.store(out_ptr, z)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold
        B,                # int32
        S,                # int32
        F,                # int32
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
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)

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

        BLOCK_SIZE_RED = 1024
        grid_red = (B * S,)
        reduce_row_sum_sumsq_kernel[grid_red](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per (batch, seq) on host (no tensor math on host)
        mean = sums / float(F)
        var = sums2 / float(F) - mean * mean
        var = torch.clamp(var, min=0.0)  # numerical safety
        std = torch.sqrt(var)

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton
        # Prepare device scalar for target_sparsity and output buffer
        target_dev = torch.tensor(float(target_sparsity), dtype=torch.float32, device=device)
        threshold_scale_buf = torch.empty(1, dtype=torch.float32, device=device)

        A1 = 1.0
        A2 = 1.133141667013566
        A3 = -2.053134318345193
        A4 = 1.343238453235822
        A5 = -0.137105471312638
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

        inv_erf_kernel[(1,)](
            threshold_scale_buf, target_dev,
            A1, A2, A3, A4, A5,
            B1, B2, B3, B4, B5,
            C1, C2, C3, C4, C5, C6,
            D1, D2, D3, D4,
        )
        threshold_scale = float(threshold_scale_buf.item())

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
