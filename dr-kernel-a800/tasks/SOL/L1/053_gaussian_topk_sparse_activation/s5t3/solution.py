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
    def reduce_rows_sum_sumsq_kernel(
        x_ptr,            # *const float32
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B, S, F,          # int32, runtime
        BLOCK_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles up to BLOCK_ROWS rows
        pid = tl.program_id(axis=0)
        row_start = pid * BLOCK_ROWS
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        mask_rows = rows < (B * S)

        b = rows // S
        s = rows % S

        # Local accumulators per row
        local_sum = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        local_sumsq = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

        # Iterate over feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            col = offs + tl.arange(0, BLOCK_SIZE)
            mask_col = col < F
            # 2D pointer: [BLOCK_ROWS, BLOCK_SIZE]
            ptrs = x_ptr + b[:, None] * (S * F) + s[:, None] * F + col[None, :]
            load_mask = mask_rows[:, None] & mask_col[None, :]
            vals = tl.load(ptrs, mask=load_mask, other=0.0)
            # Reduce across columns for each row
            local_sum += tl.sum(vals, axis=1)
            local_sumsq += tl.sum(vals * vals, axis=1)

        # Store per-row results
        tl.store(sums_ptr + rows, local_sum, mask=mask_rows)
        tl.store(sums2_ptr + rows, local_sumsq, mask=mask_rows)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F: tl.constexpr,  # feature size
    ):
        pid = tl.program_id(axis=0)
        # One program per (b, s) row
        total = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)

        mean = total / F
        var = sumsq / F - mean * mean
        # Clamp variance to non-negative to avoid tiny negatives due to fp error
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_scalar_kernel(
        p,                # scalar float32 in (0, 1)
        out_ptr,          # *float32, single element
        # Constants for Abramowitz and Stegun 7.1.26 (central region approximation)
        a1: tl.constexpr, a2: tl.constexpr, a3: tl.constexpr, a4: tl.constexpr, a5: tl.constexpr, a6: tl.constexpr,
        b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Single program computes scalar using central region approximation
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        res = poly * q / den
        tl.store(out_ptr, res)

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

        # Map linear index to (b, s, f)
        SF = S * F
        b = offs // SF
        rem = offs % SF
        s = rem // F
        f = rem % F

        x_ptrs = x_ptr + b * SF + s * F + f
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        # Gather mean and std for this (b, s)
        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        # Compute adaptive threshold: mean + std * ndtri(target_sparsity) where ndtri provided via threshold_scale
        threshold = mean + std * threshold_scale

        # Apply sparsity: y = max(0, x - threshold)
        y = x_vals - threshold
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

        # Launch reduction kernel: each program handles up to BLOCK_ROWS rows
        BLOCK_ROWS = 32
        grid_reduce = (triton.cdiv(B * S, BLOCK_ROWS),)
        reduce_rows_sum_sumsq_kernel[grid_reduce](
            x, sums, sums2, B, S, F,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_SIZE=256,  # feature chunk size
        )

        # 2) Compute mean and std per (b, s) using Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_stats = (B * S,)
        compute_mean_std_kernel[grid_stats](
            sums, sums2, mean, std, F=F
        )

        # 3) Compute ndtri(target_sparsity) scalar using Triton approximation
        # We use the central region approximation from Abramowitz & Stegun 7.1.26.
        a1 = -3.969683028665376e+01
        a2 = 2.209460984245205e+02
        a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02
        a5 = -3.066479806614716e+01
        a6 = 2.506628277459239e+00

        b1 = -5.447609879822406e+01
        b2 = 1.615858368580409e+02
        b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01
        b5 = -1.328068155288572e+01

        out = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_scalar_kernel[(1,)](
            float(target_sparsity), out,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            BLOCK_SIZE=1,
        )
        std_multiplier = float(out.item())

        # 4) Apply sparsity in Triton: output = max(0, x - (mean + std * std_multiplier))
        out_fp32 = torch.empty_like(x, dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, std_multiplier,
            B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # 5) Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
