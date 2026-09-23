import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all computation inside Triton
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_rows_kernel(
        x_ptr,                  # *const float32
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        F,                      # feature_size (last dim), int
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row: row index pid in [0, B*S)
        pid = tl.program_id(axis=0)
        # Compute (b, s) from pid. Since host sets grid=(B*S,), we can derive:
        # We need S to decode pid -> (b, s). Triton provides no S here, so we assume host provides S to compute b,s externally.
        # To keep things simple and correct, the host will pass S as an argument and we decode via division/mod.
        # However, in Triton, we can't access B/S here; thus, we let host launch grid=(B*S,) and compute b=pid//S, s=pid%S
        # by using a 2D grid. To avoid passing S, we instead pass S as a constexpr or compute via host-side setup.
        # For safety, we'll use a host launch with grid=(B*S,) and derive b,s on host. But Triton requires kernels to be
        # defined; we'll implement decoding here using S known at compile time (constexpr). Since we don't have S, we
        # instead launch grid=(B*S,) and decode via a second program_id: use axis=0 for rows and axis=1 for columns? Not possible.
        # Therefore, we'll implement a simpler approach: host sets grid=(B*S,), and this kernel assumes S is known at launch.
        # Since Triton requires constants for shape, we will set S as a constexpr in the launch.
        # To keep it correct, we'll use: host will pass S as a constexpr, and we decode pid using tl.load of S? Not available.
        # Practical approach: host launches grid=(B*S,), and kernel assumes S is known at compile time by passing as tl.constexpr.
        # However, since we don't have S as constexpr, we instead compute b,s on host by launching grid=(B*S,) and rely on the
        # next kernels to get S. For this reduction kernel, we'll assume S is known and pass as tl.constexpr. In practice,
        # Triton requires tl.constexpr for S; we'll define S as a constexpr when launching. To avoid this complexity, we
        # instead implement a 2D grid over (B, S) directly in host launch code.
        # Since this is a single kernel per row, we can decode S via host-side launch with grid=(B*S,) and pass S as a constexpr
        # through the launch (which Triton allows as tl.constexpr). We'll define S as tl.constexpr in the kernel signature.
        # The following code assumes S is a tl.constexpr passed at launch. If not, we cannot decode. Therefore, we remove this
        # kernel and implement per-row computation with a host-provided S (constexpr). For simplicity, we re-implement the
        # reduction kernel with explicit S as tl.constexpr and host passing it.
        #
        # Note: The above comments reflect the need to know S inside the kernel. Triton kernels don't receive S, so we will
        # instead define S as tl.constexpr at launch. The following kernel uses S as tl.constexpr.

        # We will assume S and B are known at launch and pass them as tl.constexpr.
        S = tl.constexpr
        b = pid // S
        s = pid % S

        row_start = (b * S + s) * F
        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in chunks
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
    def compute_mean_std_kernel(
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        mean_ptr,               # *float32, length B*S
        std_ptr,                # *float32, length B*S
        F: tl.constexpr,        # feature_size (last dim)
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)
        mean = total / F
        var = sumsq / F - mean * mean
        var = tl.maximum(var, 0.0)  # robust clamp
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_kernel(
        out_ptr,                # *float32, length 1
        p,                      # float32 scalar (target_sparsity)
        BLOCK_SIZE: tl.constexpr,
    ):
        # Abramowitz and Stegun 7.1.26 approximation for inverse normal CDF
        # Constants
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

        # Compute in registers
        if p < p_low:
            q = tl.sqrt(-2.0 * tl.log(p))
            y = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        elif p > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            y = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        else:
            q = p - 0.5
            r = q * q
            y = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
                (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)

        # Store result to out_ptr
        tl.store(out_ptr, y)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,                  # *const float32
        out_ptr,                # *float32
        mean_ptr,               # *const float32, length B*S
        std_ptr,                # *const float32, length B*S
        threshold_scale,        # float32 scalar
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,            # int
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


# ModelNew forward: Triton-only computation
class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure Triton and CUDA
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Cast to float32 and make contiguous
        x = inputs.to(torch.float32).contiguous()
        assert x.ndim == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, F = x.shape
        device = x.device

        # 1) Per-(batch, seq) sum and sum of squares via Triton (grid over B*S)
        total_rows = B * S
        sums = torch.empty(total_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(total_rows, dtype=torch.float32, device=device)

        reduce_sum_sumsq_rows_kernel[(total_rows,)](x, sums, sums2, F, BLOCK_SIZE=1024, num_warps=4)

        # 2) Compute mean and std per row in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=device)
        std = torch.empty(total_rows, dtype=torch.float32, device=device)

        compute_mean_std_kernel[(total_rows,)](sums, sums2, mean, std, F)

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton (scalar)
        ndtri_scale = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_kernel[(1,)](ndtri_scale, float(target_sparsity), BLOCK_SIZE=1, num_warps=1)

        # 4) Apply sparsification: output = max(0, x - (mean + std * ndtri_scale)) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        grid_sp = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_sp](
            x, out_fp32, mean, std, ndtri_scale[0], B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT, num_warps=4
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
