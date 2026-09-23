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
        B,                # int32
        S,                # int32
        F,                # int32 (feature dimension)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S

        # Row start in flattened [B, S, F] layout
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

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def ndtri_kernel(
        out_ptr,          # *float32, scalar output
        p,                # scalar float32 (target_sparsity)
        p_low,            # scalar float32 constant 0.02425
        p_high,           # scalar float32 constant (1 - p_low)
        # coefficients for approximations
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5,
        c1, c2, c3, c4, c5, c6,
        d1, d2, d3, d4,
    ):
        # Abramowitz and Stegun 7.1.26 approximation for normal PPF
        pl = p_low
        ph = p_high

        # lower region
        mask_low = p < pl
        q_low = tl.sqrt(-2.0 * tl.log(p))
        poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
        den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
        y_low = poly_low / den_low

        # central region
        mask_mid = (p >= pl) & (p <= ph)
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
        den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
        y_mid = poly_mid * q_mid / den_mid

        # upper region
        mask_high = p > ph
        q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
        den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
        y_high = -poly_high / den_high

        # select
        y = tl.where(mask_low, y_low, 0.0)
        y = tl.where(mask_mid, y_mid, y)
        y = tl.where(mask_high, y_high, y)

        tl.store(out_ptr, y)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32
        out_ptr,          # *const float32 (we'll write bf16-compatible values)
        mean_ptr,         # *const float32, length B*S
        std_ptr,          # *const float32, length B*S
        threshold_scale,  # scalar float32 multiplier (ndtri(target_sparsity))
        B,                # int32
        S,                # int32
        F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # 1D grid over all elements
        pid = tl.program_id(axis=0)
        total = B * S * F
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total

        SF = S * F
        b = offs // SF
        rem = offs % SF
        s = rem // F
        f = rem % F

        x_ptrs = x_ptr + offs
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        threshold = mean + std * threshold_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)

        # store as bfloat16 (Triton will cast)
        y_bf16 = y.to(tl.bfloat16)

        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, y_bf16, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only path
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 contiguous for Triton kernels
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 1024
        grid = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid](x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED)

        # 2) Compute mean and std per (b, s) on host (lightweight)
        mean = sums / float(F)
        var = sums2 / float(F) - mean * mean
        var = torch.clamp(var, min=0.0)  # guard against tiny negatives
        std = torch.sqrt(var)  # float32 tensors

        # 3) Compute inverse normal CDF for target_sparsity in Triton (scalar)
        p_low = 0.0


def run(*args):
    return ModelNew()(*args)
