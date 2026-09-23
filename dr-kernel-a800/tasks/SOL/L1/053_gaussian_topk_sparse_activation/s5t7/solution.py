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
        B: tl.constexpr,  # batch size (meta)
        S: tl.constexpr,  # seq_len (meta)
        F,                # feature_size (runtime int)
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Iterate over feature dimension
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
        F,                # feature_size (runtime int)
    ):
        pid = tl.program_id(axis=0)
        sum_val = tl.load(sums_ptr + pid)
        sumsq_val = tl.load(sums2_ptr + pid)

        mean = sum_val / F
        var = sumsq_val / F - mean * mean
        var = tl.maximum(var, 0.0)  # numerical stability
        std = tl.sqrt(var)

        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)


    @triton.jit
    def ndtri_kernel(
        out_ptr,          # *float32, length 1
        target_p,         # scalar float32
    ):
        # Abramowitz and Stegun 7.1.26 approximation for inv-Normal CDF
        p = target_p
        a1 = -3.9696830e+01
        a2 = 2.2094609e+02
        a3 = -2.7592851e+02
        a4 = 1.3835775e+02
        a5 = -3.0664798e+01
        a6 = 2.5066283e+00

        b1 = -5.4476099e+01
        b2 = 1.6158584e+02
        b3 = -1.5569898e+02
        b4 = 6.6801312e+01
        b5 = -1.3280681e+01

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

        if p <= 0.5:
            t = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = inv
        else:
            t = tl.sqrt(-2.0 * tl.log(p))
            poly = (c1 * t + c2) * t + c3
            poly = (poly * t + c4) * t + c5
            poly = (poly * t + c6)
            y = 1.0 - poly * t * tl.exp(-(t * t))
            inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
            result = -inv

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

        # Work in float32 for statistics; ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_reduce = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](x, sums, sums2, B=B, S=S, F=F)

        # 2) Compute mean and std per (b, s)
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](sums, sums2, mean, std, F)

        # 3) Compute inverse normal CDF multiplier (ndtri(target_sparsity)) in Triton
        ndtri_out = torch.empty(1, dtype=torch.float32, device=device)
        grid_ndtri = (1,)
        ndtri_kernel[grid_ndtri](ndtri_out, target_sparsity)

        # Read scalar multiplier on device
        threshold_scale = ndtri_out[0].item()

        # 4) Sparsify: output = max(0, x - (mean + std * threshold_scale)) in Triton
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
