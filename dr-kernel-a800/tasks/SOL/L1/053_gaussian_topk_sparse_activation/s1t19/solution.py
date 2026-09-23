import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    start = 0
    while start < D:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        base = pid * D  # int32 base offset into flattened [B*S, D]
        x = tl.load(X_ptr + base + offs.to(tl.int64), mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32
):
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def erfinv_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Compute erfinv(p) using a good approximation:
    # Let t = 1 - 2p, then z = erfinv(t) = sqrt(2) * erfinv(t) with erf(z) ~ t
    # Use Abramowitz & Stegun 7.1.26 for erf approximation and Newton iterations.
    p = tl.load(P_ptr)
    t = 1.0 - 2.0 * p
    # Initial guess for z = erfinv(t)
    # erf(0) = 0 for t > 0, near t for small t; we use z0 = sign(t) * sqrt(2/pi) * |t|^(1/2) for better accuracy
    pi = 3.141592653589793
    sqrt2 = 1.4142135623730951
    sqrt2_over_pi = sqrt2 / pi
    # sign
    sign = tl.where(t >= 0.0, 1.0, -1.0)
    z = sign * tl.sqrt(2.0 * tl.abs(t) * sqrt2_over_pi)
    # Newton iterations: erf(z) + t = 0
    # erf(z) approximation: asin(sqrt(1 - exp(-z^2 * 4/pi))) / sqrt(pi), but using simpler polynomial for speed
    # We'll use 8 iterations for robustness.
    # Note: Triton does not provide math.erf directly; implement using a polynomial approximation.
    # We approximate erf(z) with the Maclaurin series terms up to z^7 and update z.
    # For simplicity and robustness, use a fixed number of iterations and a decent initial guess.
    # Use erf(z) ≈ (2/√π)z * (1 + a1 z^2 + a2 z^4 + a3 z^6), then update z via Newton.
    # Coefficients:
    a1 = 0.074760788
    a2 = 0.074760788 * 0.074760788 * 4.0
    a3 = 0.074760788 * 0.074760788 * 0.074760788 * 8.0
    # Perform 8 iterations
    for _ in range(8):
        z2 = z * z
        # erf(z) ≈ (2/√π) * z * (1 + a1 z^2 + a2 z^4 + a3 z^6)
        erf_z = (2.0 / tl.sqrt(pi)) * z * (1.0 + a1 * z2 + a2 * (z2 * z2) + a3 * (z2 * z2 * z2))
        # Update: z_{n+1} = z_n + (erf(z_n) + t) / (2/sqrt(pi) * (1 + 3a1 z_n^2 + 15a2 z_n^4 + 105a3 z_n^6))
        denom = (2.0 / tl.sqrt(pi)) * (1.0 + 3.0 * a1 * z2 + 15.0 * a2 * (z2 * z2) + 105.0 * a3 * (z2 * z2 * z2))
        z = z + (erf_z + t) / denom
    # Store z = erfinv(t) = erfinv(1 - 2p)
    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bfloat16, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid over (B, S, tiles of D)
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # int32

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar float32
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
    y = x - threshold  # broadcast scalar threshold
    y = tl.maximum(y, 0.0)  # ReLU
    # Store as bfloat16
    tl.store(OUT_ptr + base_idx + offs.to(tl.int64), y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation is Triton kernels

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are contiguous and on device
        assert inputs.is_cuda, "ModelNew.forward expects CUDA tensors"
        inputs = inputs.contiguous()

        B, S, D = inputs.shape
        # Buffers for sums, sumsq, mean, std, and z_score
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Scalar z-score buffer (length 1)
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s)
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf, B, S, D,
            BLOCK_SIZE=1024, num_warps=8
        )

        # Launch compute mean/std kernel: one program per (b, s)
        grid_mstd = (B * S,)
        compute_mean_std_kernel[grid_mstd](
            sum_buf, sumsq_buf, mean_buf, std_buf, D,
            num_warps=1
        )

        # Launch ndtri (erfinv-based) kernel: scalar
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        erfinv_approx_kernel[(1,)](
            p_tensor, z_buf,
            num_warps=1
        )

        # Prepare output as bfloat16 and apply activation
        out = torch.empty((B, S, D), dtype=torch.bfloat16, device=inputs.device)

        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_buf, out, B, S, D,
            BLOCK_SIZE=1024, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
