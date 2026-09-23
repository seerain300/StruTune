import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean across the last dimension (features).
@triton.jit
def mean_lastdim_kernel(
    x_ptr,            # *fp32, shape [NROWS, F] contiguous
    mean_out_ptr,     # *fp32, shape [NROWS]
    F: tl.constexpr,  # number of features
    NROWS: tl.constexpr,  # number of rows = B*S
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 <= pid < NROWS
    total = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + pid * F + idx, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    mean = total / F
    tl.store(mean_out_ptr + pid, mean)


# Triton kernel: compute per-row population std across the last dimension (features).
@triton.jit
def std_lastdim_kernel(
    x_ptr,            # *fp32, shape [NROWS, F] contiguous
    std_out_ptr,      # *fp32, shape [NROWS]
    mean_ptr,         # *fp32, shape [NROWS]
    F: tl.constexpr,  # number of features
    NROWS: tl.constexpr,  # number of rows = B*S
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 <= pid < NROWS
    mean_val = tl.load(mean_ptr + pid)
    sum_sq = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + pid * F + idx, mask=mask, other=0.0)
        diff = vals - mean_val
        sum_sq += tl.sum(diff * diff, axis=0)
    std = tl.sqrt(sum_sq / F)
    tl.store(std_out_ptr + pid, std)


# Triton kernel: compute inverse-normal CDF (ndtri) for scalar p using A&S 5.2.23.
@triton.jit
def ndtri_kernel(
    z_ptr,        # *fp32, shape [1] (scalar output)
    p: tl.float32,
    p_low: tl.float32,
    p_high: tl.float32,
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
):
    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    cond_low = p < p_low
    cond_up = p > p_high
    z_val = tl.where(cond_low, z_low, tl.where(cond_up, z_up, z_mid))

    tl.store(z_ptr, z_val)


# Triton kernel: apply cutoff and ReLU for each row across features.
@triton.jit
def apply_cutoff_relu_kernel(
    x_ptr,           # *fp32, shape [NROWS, F] contiguous
    mean_ptr,        # *fp32, shape [NROWS]
    std_ptr,         # *fp32, shape [NROWS]
    z_ptr,           # *fp32, shape [1] (scalar z)
    out_ptr,         # *fp32, shape [NROWS, F] contiguous
    F: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 <= pid < NROWS
    mean_val = tl.load(mean_ptr + pid)
    std_val = tl.load(std_ptr + pid)
    z_val = tl.load(z_ptr)  # scalar float32
    cutoff = mean_val + std_val * z_val
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + pid * F + idx, mask=mask, other=0.0)
        y = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + pid * F + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.5):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation of the original run function.

        Args:
            inputs: Tensor of shape [batch_size, seq_len, intermediate_size]
        Returns:
            Tensor of same shape as input, sparsified via ReLU(x - cutoff) with
            cutoff = mean + std * ndtri(target_sparsity).
        """
        assert inputs.dim() == 3, "inputs must be [B, S, F]"
        assert inputs.is_cuda, "inputs must be on CUDA device for Triton"

        B, S, F = inputs.shape
        inputs_f32 = inputs.to(torch.float32).contiguous()

        # View as [NROWS, F] where NROWS = B*S
        NROWS = B * S
        x_2d = inputs_f32.view(NROWS, F).contiguous()

        # Allocate outputs and buffers
        mean_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)
        std_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)

        # Constants for ndtri approximation (Abramowitz & Stegun 5.2.23)
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

        # 1) Compute mean per row
        mean_lastdim_kernel[(NROWS,)](
            x_2d, mean_out, F, NROWS, BLOCK_F=2048, num_warps=4
        )

        # 2) Compute std per row (population std)
        std_lastdim_kernel[(NROWS,)](
            x_2d, std_out, mean_out, F, NROWS, BLOCK_F=2048, num_warps=4
        )

        # 3) Compute scalar z = ndtri(target_sparsity) inside Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        ndtri_kernel[(1,)](
            z_buf,
            self.target_sparsity,
            p_low, p_high,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            num_warps=1
        )
        # Read z once; minimal host-side interaction
        z_val = float(z_buf.item())

        # 4) Apply cutoff and ReLU in Triton, write to out_fp32
        out_fp32 = torch.empty_like(x_2d, dtype=torch.float32, device=inputs.device)
        apply_cutoff_relu_kernel[(NROWS,)](
            x_2d, mean_out, std_out, z_buf, out_fp32, F, BLOCK_F=2048, num_warps=4
        )

        # 5) Reshape back to [B, S, F] and cast to original dtype
        out_3d = out_fp32.view(B, S, F)
        return out_3d.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
