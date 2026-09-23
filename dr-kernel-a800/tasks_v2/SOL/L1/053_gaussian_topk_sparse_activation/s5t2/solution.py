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
    def reduce_sum_sumsq_kernel(
        x_ptr,                  # *const float32, flattened input
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        B, S, F,                # int32, dims
        BLOCK_SIZE: tl.constexpr,
    ):
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

        tl.atomic_add(sums_ptr + pid, local_sum)
        tl.atomic_add(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        mean_ptr,               # *float32, length B*S
        std_ptr,                # *float32, length B*S
        F: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        total2 = tl.load(sums2_ptr + pid)
        mean = total / F
        var = total2 / F - mean * mean
        var = tl.maximum(var, 0.0)  # guard against tiny negative due to numerical error
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_scalar_kernel(
        out_ptr,                # *float32, scalar output for ndtri(target_sparsity)
        p,                      # float32, target sparsity
    ):
        # Abramowitz & Stegun 7.1.26 approximation for the standard normal quantile
        # We pass a scalar p (target_sparsity) and compute z = ndtri(p).
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

        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00
        c6 = 2.938163982698783e+00

        d1 = 7.784695709041462e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00

        p_low = 0.02425
        p_high = 1.0 - p_low

        # Compute p_scalar as float
        p_scalar = p  # scalar input
        if p_scalar < p_low:
            z = tl.sqrt(-2.0 * tl.log(p_scalar))
            num = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
            den = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
            z = num / den
        elif p_scalar > p_high:
            z = tl.sqrt(-2.0 * tl.log(1.0 - p_scalar))
            num = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
            den = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
            z = num / den
        else:
            q = p_scalar - 0.5
            r = q * q
            num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            z = num * q / den

        tl.store(out_ptr, z)

    @triton.jit
    def sparsify_kernel(
        x_ptr,                  # *const float32 (input)
        out_ptr,                # *float32 (output; we will cast to bfloat16 after)
        threshold_ptr,          # *const float32, shape [B*S]
        total_elems,            # int32, total number of elements
        B, S, F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        start = pid * BLOCK_SIZE
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elems

        # Compute (b, s) for each element
        B_dim = S * F
        s = idx // B_dim
        b = (idx // F) % B
        bs = b * S + s

        # Gather input
        in_ptrs = x_ptr + idx
        x_vals = tl.load(in_ptrs, mask=mask, other=0.0)

        # Gather threshold for each (b, s)
        th_ptrs = threshold_ptr + bs
        th_vals = tl.load(th_ptrs, mask=mask, other=0.0)  # scalar per element

        diff = x_vals - th_vals
        # ReLU
        diff = tl.maximum(diff, 0.0)

        # Store float32 output
        out_ptrs = out_ptr + idx
        tl.store(out_ptrs, diff, mask=mask)


# -----------------------------
# Entry point module
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Default target sparsity; can be set externally
        self.target_sparsity = 0.5

    def forward(self, *args):
        # Expect single tensor input: [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor input of shape [batch_size, seq_len, intermediate_size].")
        inputs = args[0]

        # If Triton not available or not CUDA, fallback to original PyTorch behavior
        if not TRITON_AVAILABLE or not inputs.is_cuda:
            x = inputs.to(torch.float32)
            B, S, F = x.shape
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
            # Compute normal inverse CDF multiplier using torch.special.erfinv (host-only minimal math)
            try:
                from torch.special import erfinv
                z = torch.sqrt(torch.tensor(2.0, device=x.device)) * erfinv((2.0 * self.target_sparsity - 1.0).to(x.device))
                multiplier = float(z.item())
            except Exception:
                multiplier = 0.0
            cutoff = mean + std * multiplier
            sparse = torch.relu(x - cutoff)
            return sparse.to(torch.bfloat16)

        # Triton path: ensure contiguous and float32 for stats
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        total_rows = B * S
        total_elems = B * S * F

        # 1) Reduce per-row sums and sums of squares
        sums = torch.zeros(total_rows, dtype=torch.float32, device=x.device)
        sums2 = torch.zeros(total_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = 1024
        reduce_sum_sumsq_kernel[(total_rows,)](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE
        )

        # 2) Compute mean and std per (b, s) in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        compute_mean_std_kernel[(total_rows,)](
            sums, sums2, mean, std, F=F
        )
        mean = mean.view(B, S)  # [B, S]
        std = std.view(B, S)    # [B, S]

        # 3) Compute threshold multiplier using Triton (Abramowitz & Stegun approximation)
        multiplier_out = torch.empty((), dtype=torch.float32, device=x.device)
        ndtri_scalar_kernel[(1,)](
            multiplier_out, self.target_sparsity
        )
        multiplier = float(multiplier_out.item())  # scalar

        # 4) Build threshold tensor [B, S, 1]
        cutoff = mean + std * multiplier  # [B, S]
        threshold = cutoff.unsqueeze(-1).contiguous()  # [B, S, 1]

        # 5) Apply sparsification: output = max(0, x - threshold) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        BLOCK_SIZE_POINT = 1024
        grid = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_kernel[grid](
            x, out_fp32, threshold, total_elems, B, S, F, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
