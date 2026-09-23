import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Accumulate sum and sum of squares using BLOCKed reduction over H
    sum_val = 0.0
    sum_sq = 0.0

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)

    for j in range(0, H, BLOCK):
        idx = j + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = H  # population count
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar_hastings(out_ptr, p_val):
    # Compute inverse standard normal CDF for p in [0.5, 1) using Hastings approximation
    # We compute for p = 1 - target_sparsity, which lies in [0.5, 1).
    # Hastings formula:
    # t = 1 - p
    # x = sqrt(-ln(t))
    # d = 1 / (1 + p)
    # z = sign*(x + (a1*d + a2*d^2 + a3*d^3 + a4*d^4))
    # Coefficients
    a1 = 1.2655125232874789
    a2 = 1.0000236813195244
    a3 = 0.37409196421638332
    a4 = 0.09678417646783182

    # p is in [0.5, 1), so sign = 1
    t = 1.0 - p_val
    x = tl.sqrt(-tl.log(t))  # natural log
    p = 1.0 - t  # since t = 1 - p, and we take p in [0.5, 1)
    d = 1.0 / (1.0 + p)
    poly = a1 * d + a2 * (d * d) + a3 * (d * d * d) + a4 * (d * d * d * d)
    nd = x + poly
    # Since p in [0.5, 1) => nd >= 0, no sign adjustment needed
    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32 scalar per row

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)

    # Elementwise apply: y = max(x - thr, 0), store as bfloat16
    for j in range(0, H, BLOCK):
        idx = j + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle trivial cases: no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous 3D tensor
        if inputs.ndim != 3:
            raise ValueError("inputs must be 3D tensor of shape [B, S, H]")
        inputs = inputs.contiguous()

        B, S, H = inputs.shape
        device = inputs.device

        # Allocate per-row mean and std (fp32)
        mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # 1) Compute mean and std with Triton
        BLOCK = 1024
        compute_mean_std_fp32[(B * S,)](
            inputs, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # 2) Compute icdf(z) for p = 1 - target_sparsity using Triton
        # p is in [0.5, 1) because target_sparsity in (0, 1)
        p_val = 1.0 - target_sparsity
        z_out = torch.empty((1,), dtype=torch.float32, device=device)
        compute_icdf_scalar_hastings[(1,)](
            z_out, p_val,
            num_warps=1,
        )

        # 3) Apply threshold and ReLU, write output as bfloat16
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)
        apply_threshold_relu_to_bf16[(B * S,)](
            inputs, out, mean, std, B, S, H, z_out,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
