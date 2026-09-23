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
        B: tl.constexpr,  # batch size (constexpr)
        S: tl.constexpr,  # seq_len (constexpr)
        F,                # feature_size (runtime int)
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        s = pid % S
        b = pid // S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension F
        for i in range(0, F):
            val = tl.load(x_ptr + row_start + i)
            local_sum += val
            local_sumsq += val * val

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,         # *const float32, length B*S
        sums2_ptr,        # *const float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F,                # int
    ):
        pid = tl.program_id(axis=0)  # 0..(B*S-1)
        sum = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)
        mean = sum / F
        var = sumsq / F - mean * mean
        var = tl.maximum(var, 0.0)  # clamp to avoid tiny negative due to roundoff
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_kernel(
        out_ptr,          # *float32, length 1
        p,                # float32 scalar target_sparsity
    ):
        # Abramowitz and Stegun 7.1.26 approximation (erfinv-based)
        # For p in (0,1), result is positive when p <= 0.5, negative when p > 0.5
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

        ax = tl.abs(p)
        if ax <= 0.5:
            t = tl.sqrt(-2.0 * tl.log(1.0 - ax))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = inv  # positive branch
        else:
            t = tl.sqrt(-2.0 * tl.log(ax))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = -inv  # negative branch

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

        # Convert to float32 for statistics and ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_reduce = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B=B, S=S, F=F
        )

        # 2) Compute mean and std per (batch, seq) in Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_meanstd = (B * S,)
        compute_mean_std_kernel[grid_meanstd](
            sums, sums2, mean, std, F
        )

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton -> scalar
        ndtri_out = torch.empty(1, dtype=torch.float32, device=device)
        grid_ndtri = (1,)
        ndtri_kernel[grid_ndtri](ndtri_out, target_sparsity)
        threshold_scale = ndtri_out[0].item()  # scalar multiplier

        # 4) Apply sparsity: output = max(0, x - (mean + std * threshold_scale)) in Triton
        total_elems = B * S * F
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        BLOCK_SIZE = 1024
        grid_sparsify = (triton.cdiv(total_elems, BLOCK_SIZE),)
        sparsify_relu_kernel[grid_sparsify](
            x, out_fp32, mean, std, threshold_scale,
            B=B, S=S, F=F, total_elems=total_elems, BLOCK_SIZE=BLOCK_SIZE
        )

        # Return as bfloat16 to match the original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
