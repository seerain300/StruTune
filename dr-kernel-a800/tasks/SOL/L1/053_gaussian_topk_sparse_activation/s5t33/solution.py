import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def reduce_row_sum_sumsq_kernel(
        x_ptr,            # *const float32, flattened input [B*S*F]
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        F,                # int32, feature size
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        row_start = pid * F

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
    def compute_mean_std_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F: tl.constexpr,  # feature size
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        total2 = tl.load(sums2_ptr + pid)
        mean = total / F
        var = total2 / F - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def inv_erf_kernel(
        out_ptr,          # *float32, length 1 (scalar)
        p_ptr,            # *const float32, length 1 (scalar input p)
        A1, A2, A3, A4, A5,
        B1, B2, B3, B4, B5,
        C1, C2, C3, C4, C5, C6,
        D1, D2, D3, D4,
    ):
        # Compute inverse erf approximation (Abramowitz & Stegun 7.1.26) for a single scalar p
        p = tl.load(p_ptr)
        # t = 1 / (1 + p * (A1 + p * (A2 + p * (A3 + p * (A4 + p * A5)))))
        t = 1.0 / (1.0 + p * (A1 + p * (A2 + p * (A3 + p * (A4 + p * A5)))))
        # poly1 = (((C1*t + C2)*t + C3)*t + C4)*t + C5)*t + C6
        # poly2 = (((D1*t + D2)*t + D3)*t + D4)*t + 1.0
        poly1 = (((C1 * t + C2) * t + C3) * t + C4) * t + C5
        poly2 = (((D1 * t + D2) * t + D3) * t + D4) * t + 1.0
        x = 1.0 - p * poly1 / poly2
        # Store result
        tl.store(out_ptr, x)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32, input [B*S*F]
        out_ptr,          # *float32, output [B*S*F]
        mean_ptr,         # *const float32, [B*S]
        std_ptr,          # *const float32, [B*S]
        inv_scale_ptr,    # *const float32, length 1 (scalar inv-erf(target_sparsity) * sqrt(2))
        total_elems,      # int32
        B: tl.constexpr,  # batch size
        S: tl.constexpr,  # seq_len
        F: tl.constexpr,  # feature size
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

        x_ptrs = x_ptr + offs
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        inv_scale = tl.load(inv_scale_ptr)  # scalar

        threshold = mean + std * inv_scale
        y = x_vals - threshold
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)

        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure contiguous and compute in float32 for stability
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_elems = B * S * F

        device = x.device

        # 1) Per-(batch, seq) sums and sums of squares (reduce over feature dim)
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_red = (B * S,)
        reduce_row_sum_sumsq_kernel[grid_red](x, sums, sums2, F, BLOCK_SIZE=1024)

        # 2) Compute mean and std per row in Triton (compute_mean_std_kernel)
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_stats = (B * S,)
        compute_mean_std_kernel[grid_stats](sums, sums2, mean, std, F)

        # 3) Compute inv-erf(target_sparsity) * sqrt(2) in Triton (scalar)
        # Abramowitz & Stegun 7.1.26 constants
        A1 = 1.005399895013267
        A2 = 0.023198394241822
        A3 = -0.000728634698644
        A4 = 0.000002394864317
        A5 = -0.000000003951815
        B1 = -0.019194201964488
        B2 = 0.000499643785993
        B3 = -0.000001480350212
        B4 = 0.000000002365616
        B5 = -0.000000000002104
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

        # Pass target_sparsity as a device scalar to Triton
        p_tensor = torch.tensor(float(target_sparsity), dtype=torch.float32, device=device)  # single scalar
        inv_scale_buf = torch.empty(1, dtype=torch.float32, device=device)
        inv_erf_kernel[(1,)](
            inv_scale_buf, p_tensor,
            A1, A2, A3, A4, A5,
            B1, B2, B3, B4, B5,
            C1, C2, C3, C4, C5, C6,
            D1, D2, D3, D4,
        )
        # Multiply by sqrt(2) to get ndtri(target_sparsity)
        inv_scale_buf.mul_(math.sqrt(2.0))

        # 4) Apply sparsity: output = max(0, x - (mean + std * inv_scale)) in Triton
        out_fp32 = torch.empty_like(x, dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, inv_scale_buf, total_elems, B, S, F, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
