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
        B: tl.constexpr,  # batch size
        S: tl.constexpr,  # seq_len
        F,                # feature_size (runtime int)
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        offs = tl.arange(0, F)
        vals = tl.load(x_ptr + row_start + offs)
        local_sum = tl.sum(vals, axis=0)
        local_sumsq = tl.sum(vals * vals, axis=0)

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
        total = tl.load(sums_ptr + pid)
        sqsum = tl.load(sums2_ptr + pid)
        mean = total / F
        var = sqsum / F - mean * mean
        var = tl.maximum(var, 0.0)  # clamp for numerical stability
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)


    @triton.jit
    def erfinv_approx_kernel(
        x_ptr,            # *float32, 1-element tensor containing target_sparsity
        out_ptr,          # *float32, 1-element output tensor for ndtri
        a1 = -1.3065426083333333,
        a2 =  7.746764913333333,
        a3 = -33.023303166666666,
        a4 = 69.30482433333333,
        a5 = -62.0886096,
        a6 = 25.6858232,
        p = 0.3275911,
    ):
        x = tl.load(x_ptr)
        # Abramowitz and Stegun 7.1.26: erf(z) approximation and erfinv
        sign = tl.where(x < 0.0, -1.0, 1.0)
        ax = tl.abs(x)
        t = 1.0 - ax
        poly = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6)
        y = 1.0 - poly * t * tl.exp(-(ax * ax))
        inv = 0.5 * tl.sqrt(tl.maximum(1.0 - y * y, 0.0))
        result = sign * inv
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

        # Work in float32 for statistics, ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares (


def run(*args):
    return ModelNew()(*args)
