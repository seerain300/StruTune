import torch
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Constants for the Abramowitz and Stegun approximation (as in the original code).
# We'll use this in PyTorch to compute the scalar multiplier. Triton kernel won't use this since
# we'll compute the multiplier on device using PyTorch.
A1 = -3.969683028665376e+01
A2 = 2.209460984245205e+02
A3 = -2.759285104469687e+02
A4 = 1.383577518672690e+02
A5 = -3.066479806614716e+01
A6 = 2.506628277459239e+00

B1 = -5.447609879822406e+01
B2 = 1.615858368580409e+02
B3 = -1.556989798598866e+02
B4 = 6.680131188771972e+01
B5 = -1.328068155288572e+01

C1 = -7.784894002430293e-03
C2 = -3.223964580411365e-01
C3 = -2.400758277161838e+00
C4 = -2.549732539343734e+00
C5 = 4.374664141464968e+00
C6 = 2.938163982698783e+00

D1 = 7.784695709041462e-03
D2 = 3.224671290700398e-01
D3 = 2.445134137142996e+00
D4 = 3.754408661907416e+00

P_LOW = 0.02425
P_HIGH = 1.0 - P_LOW

# Triton kernel: compute per-row mean and std along the last dimension.
# Input: X [B, S, N], outputs: mean_out [B*S], std_out [B*S]
@triton.jit
def _row_stats_kernel(
    X_ptr,            # *f32
    mean_out_ptr,     # *f32, length B*S
    std_out_ptr,      # *f32, length B*S
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    N,                # int
    BLOCK: tl.constexpr,  # chunk size along feature dimension
):
    pid = tl.program_id(0)  # 0..(B*S-1)
    row_index = pid

    # compute b, s from row_index
    b = row_index // S
    s = row_index % S

    # base offset for this row in [b, s, :]
    # Using strides: X is contiguous with stride (S*N, N, 1)
    # For [b, s, :] the starting offset is b*S*N + s*N
    base = b * S * N + s * N

    # Accumulators (scalars)
    x_sum = 0.0
    x_sumsq = 0.0

    # Loop over feature dimension in chunks
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        # x is vector, reduce to scalars
        x_sum += tl.sum(x, axis=0)
        x_sumsq += tl.sum(x * x, axis=0)
        i += BLOCK

    # Compute mean and std (population variance, unbiased=False)
    N_f = N
    mean = x_sum / N_f
    var = x_sumsq / N_f - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write out
    out_index = row_index  # linear index for [B*S]
    tl.store(mean_out_ptr + out_index, mean)
    tl.store(std_out_ptr + out_index, std)


# Triton kernel: elementwise ReLU on (inputs - threshold), per row along N.
# Inputs: Inputs [B*S, N] (flattened), threshold [B*S] (one scalar per row),
# Outputs: Out [B*S, N]
@triton.jit
def _relu_apply_kernel(
    Inputs_ptr,        # *f32, shape [B*S, N]
    Threshold_ptr,     # *f32, shape [B*S]
    Out_ptr,           # *f32, shape [B*S, N]
    B: tl.constexpr,   # int
    S: tl.constexpr,   # int
    N,                 # int
    BLOCK: tl.constexpr,
):
    row_index = tl.program_id(0)  # 0..(B*S-1)
    # base offsets
    in_base = row_index * N
    out_base = row_index * N

    # Load threshold for this row (scalar)
    thr = tl.load(Threshold_ptr + row_index)

    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(Inputs_ptr + in_base + offs, mask=mask, other=0.0)
        diff = x - thr
        relu = tl.maximum(diff, 0.0)
        tl.store(Out_ptr + out_base + offs, relu, mask=mask)
        i += BLOCK


# Helper to compute ndtri scalar on device using the approximation
def _ndtri_scalar(device, target_sparsity: float) -> torch.Tensor:
    # We keep the same approximation as in the original code. torch.tensor here is not using Triton;
    # this is acceptable as the original code does too. We compute on device.
    p = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
    # Use the same piecewise logic but scalar-only:
    low = P_LOW
    high = 1.0 - low
    q = torch.sqrt(-2.0 * torch.log(p))
    z_low = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) / \
            ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((A1 * r_mid + A2) * r_mid + A3) * r_mid + A4) * r_mid + A5) * r_mid + A6) * q_mid / \
            (((((B1 * r_mid + B2) * r_mid + B3) * r_mid + B4) * r_mid + B5) * r_mid + 1.0)
    q_high = torch.sqrt(-2.0 * torch.log(1.0 - p))
    z_high = -(((((C1 * q_high + C2) * q_high + C3) * q_high + C4) * q_high + C5) * q_high + C6) / \
             ((((D1 * q_high + D2) * q_high + D3) * q_high + D4) * q_high + 1.0)
    mask_low = p < low
    mask_mid = (p >= low) & (p <= high)
    mask_high = p > high
    # Select appropriate branch
    z = torch.zeros_like(p)
    z = torch.where(mask_low, z_low, z)
    z = torch.where(mask_mid, z_mid, z)
    z = torch.where(mask_high, z_high, z)
    return z  # shape: [1] tensor on device


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure we have a CUDA device for Triton; fallback to PyTorch if not available.
        if (not TRITON_AVAILABLE) or (inputs.device.type != "cuda"):
            # Fallback to original PyTorch behavior
            inputs_f32 = inputs.to(torch.float32)
            inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
            inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
            multiplier = _ndtri_scalar(inputs.device, target_sparsity)  # device scalar
            cutoff_threshold = inputs_mean + inputs_std * multiplier
            sparse_output = F.relu(inputs_f32 - cutoff_threshold)
            return sparse_output.to(torch.bfloat16)

        # Early return if no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Input must be 3D: [B, S, N]
        assert inputs.ndim == 3, "inputs must be of shape [batch_size, seq_len, intermediate_size]"
        B, S, N = inputs.shape

        # Compute in fp32 for numerical stability
        X = inputs.contiguous().to(torch.float32)

        # Prepare output tensors for mean and std (length B*S)
        mean_out = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_out = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel to compute per-row mean and std
        BLOCK = 256  # chunk size along feature dimension
        grid = (B * S,)
        _row_stats_kernel[grid](
            X, mean_out, std_out,
            B=B, S=S, N=N,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute scalar multiplier (inverse CDF) on device
        multiplier = _ndtri_scalar(inputs.device, target_sparsity)  # shape: [1], device
        # Compute adaptive cutoff per row: shape [B, S, 1]
        cutoff_threshold = (mean_out.view(B, S, 1) + std_out.view(B, S, 1) * multiplier)

        # Prepare flattened inputs for elementwise kernel: shape [B*S, N]
        Inputs_flat = X.reshape(B * S, N).contiguous()

        # Allocate output for elementwise computation in fp32
        Out_flat = torch.empty((B * S, N), dtype=torch.float32, device=inputs.device)

        # Launch elementwise ReLU kernel
        grid_relu = (B * S,)
        _relu_apply_kernel[grid_relu](
            Inputs_flat, cutoff_threshold.reshape(B * S).contiguous(),
            Out_flat,
            B=B, S=S, N=N,
            BLOCK=256,
            num_warps=4,
        )

        # Reshape back to [B, S, N] and cast to bf16 to match original behavior
        sparse_output = Out_flat.view(B, S, N)
        return sparse_output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
