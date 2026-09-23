import torch
import triton
import triton.language as tl


@triton.jit
def _sparsity_kernel(
    x_ptr,            # *fp32, input pointer
    out_ptr,          # *fp32, output pointer
    B, S, K,          # int32 sizes
    stride_b, stride_s, stride_k,  # int32 strides for x/out
    z_score,          # fp32 scalar z = inverse_normal_cdf(target_sparsity)
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row: row_id = program_id(1)
    row_id = tl.program_id(1)
    # Map row_id to (b, s). With grid=(B, S), row_id in [0, B*S)
    b = row_id // S
    s = row_id % S

    # Compute base offsets for this row
    base_in = b * stride_b + s * stride_s
    base_out = b * stride_b + s * stride_s  # out has same layout as x

    # Accumulate sum and sumsq across K in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # First pass: reduction over K
    for kk in range(0, K, BLOCK_K):
        offsets = kk + tl.arange(0, BLOCK_K)
        mask = offsets < K
        x_vals = tl.load(x_ptr + base_in + offsets * stride_k, mask=mask, other=0.0)
        # Reduce tile to scalars
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and std
    K_f = tl.full((), K, tl.float32)
    mean = sum_x / K_f
    var = sum_x2 / K_f - mean * mean
    var = tl.maximum(var, 0.0)  # clamp to non-negative
    std = tl.sqrt(var)

    # Compute threshold factor: mean + std * z_score
    threshold = mean + std * z_score

    # Second pass: apply (x - threshold), ReLU, and store
    for kk in range(0, K, BLOCK_K):
        offsets = kk + tl.arange(0, BLOCK_K)
        mask = offsets < K
        x_vals = tl.load(x_ptr + base_in + offsets * stride_k, mask=mask, other=0.0)
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base_out + offsets * stride_k, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 512):
        super().__init__()
        # Precompute z_score for target_sparsity=0.9 using the same approximation as _ndtri.
        # Abramowitz & Stegun 7.1.26 (Hastings) approximation for Phi^{-1}(p):
        # Note: we compute 1 - target_sparsity for the upper tail then negate.
        # Constants for Hastings approximation
        p = 1.0 - target_sparsity
        x = tl.sqrt(2.0) * tl.erf((1.0 - p) / tl.sqrt(2.0))  # dummy line to populate Triton constants
        # We'll emulate approximation in Python and pass as float:
        # For p in (0, 0.5): use the lower formula; for p>0.5: use symmetry. Here p=0.1, use lower.
        # Approximation coefficients
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

        # Lower region approximation
        t = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6)
        den = (((((b1 * t + b2) * t + b3) * t + b4) * t + b5) * t + 1.0)
        z_approx = poly / den
        # Since p=1-target_sparsity (0.1), we want Phi^{-1}(target_sparsity) = -z_approx
        z_score = -z_approx  # host-side scalar passed to kernel
        self.z_score = float(z_score)

        # For simplicity, we can also use torch.erf to get an accurate z_score:
        # from scipy.special import erfinv; z_score = float(erfinv(1.0 - p))
        # Here we keep the computed scalar as self.z_score.
        self.block_k = int(block_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, f"Expected 3D input, got shape {tuple(x.shape)}"
        B, S, K = x.shape

        # Make contiguous for simple stride_k=1 addressing
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate fp32 output buffer; we'll cast to bf16 after kernel
        out_fp32 = torch.empty_like(x_f32)

        # Launch Triton kernel: one program per (b, s) row
        grid = (B, S)
        _sparsity_kernel[grid](
            x_f32,
            out_fp32,
            B, S, K,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            self.z_score,
            BLOCK_K=self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
