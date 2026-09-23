import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per-row reduction for mean and population std along last dimension
if TRITON_AVAILABLE:
    @triton.jit
    def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
        row_id = tl.program_id(axis=0)
        base = row_id * N

        total = 0.0
        total_sq = 0.0

        for start in range(0, N, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(inp_ptr + base + offs, mask=mask, other=0.0)
            x = x.to(tl.float32)
            total += tl.sum(x, axis=0)
            total_sq += tl.sum(x * x, axis=0)

        n_float = tl.full((), N, tl.float32)
        mean = total / n_float
        var = total_sq / n_float - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        tl.store(mean_ptr + row_id, mean)
        tl.store(std_ptr + row_id, std)


    # Triton kernel: compute A&S inv_std_normal(p) and write to out_ptr[0]
    @triton.jit
    def _ndtri_scalar_kernel(p_ptr, out_ptr):
        p = tl.load(p_ptr)

        # A&S 7.1.26 constants
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

        # lower region
        q_low = tl.sqrt(-2.0 * tl.log(p))
        z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

        # central region
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

        # upper region
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
               ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

        # piecewise select
        z = tl.where(p < p_low, z_low, 0.0)
        z = tl.where(p >= p_low, z, z)
        z = tl.where(p >= p_high, z_up, z)

        tl.store(out_ptr, z)


    # Triton kernel: apply sparsity (threshold + ReLU)
    @triton.jit
    def _row_sparsify_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
        row_id = tl.program_id(axis=0)
        base_in = row_id * N
        base_out = row_id * N

        mean = tl.load(mean_ptr + row_id)
        std = tl.load(std_ptr + row_id)
        cutoff = mean + std * std_multiplier

        for start in range(0, N, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(inp_ptr + base_in + offs, mask=mask, other=0.0).to(tl.float32)
            y = x - cutoff
            y = tl.maximum(y, 0.0)  # ReLU
            tl.store(out_ptr + base_out + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version. All device tensor computations are performed in Triton kernels.
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs

        assert inputs.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, N = inputs.shape

        # Compute in float32 on device
        inp = inputs.contiguous().to(torch.float32)

        # Buffers for mean and std
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B * S,)
        BLOCK_SIZE = 1024  # reasonable chunk size for typical N up to 16k
        _row_reduce_mean_std_kernel[grid](inp, mean_buf, std_buf, N, BLOCK_SIZE)

        # Compute std_multiplier using Triton scalar kernel (no torch ops on device tensors in forward)
        p_dev = torch.tensor(float(target_sparsity), dtype=torch.float32, device=inputs.device)
        std_multiplier_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier_buf)
        std_multiplier_scalar = float(std_multiplier_buf.item())

        # Output buffer
        out = torch.empty_like(inp)

        # Launch sparsification kernel
        _row_sparsify_kernel[grid](inp, out, mean_buf, std_buf, N, std_multiplier_scalar, BLOCK_SIZE)

        # Cast back to original dtype
        return out.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
