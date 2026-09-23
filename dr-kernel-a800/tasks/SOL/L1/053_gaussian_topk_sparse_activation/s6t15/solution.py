import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_rows_kernel(
    x_ptr,            # *const float (input, cast to float32 in kernel)
    out_sum_ptr,      # *float32, shape [B*S]
    out_sumsq_ptr,    # *float32, shape [B*S]
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    F: tl.constexpr,  # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # program id: 0..(B*S-1)
    b = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_F)
    total_sum = 0.0
    total_sumsq = 0.0

    # Loop across the feature dimension in chunks of BLOCK_F
    for start in range(0, F, BLOCK_F):
        idx = start + offs
        mask = idx < F
        x_index = b * stride_b + s * stride_s + idx * stride_f
        x_vals = tl.load(x_ptr + x_index, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        total_sum += tl.sum(x_vals, axis=0)
        total_sumsq += tl.sum(x_vals * x_vals, axis=0)

    # Write per-row results
    tl.atomic_add(out_sum_ptr + pid, total_sum)
    tl.atomic_add(out_sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,       # *const float32, shape [B*S]
    out_sumsq_ptr,     # *const float32, shape [B*S]
    out_mean_ptr,      # *float32, shape [B*S]
    out_std_ptr,       # *float32, shape [B*S]
    B: tl.constexpr,   # int
    S: tl.constexpr,   # int
    F: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # 0..(B*S-1)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    # population mean and std (unbiased=False)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def apply_threshold_kernel(
    x_ptr,              # *const float (input)
    mean_ptr,           # *const float32, shape [B*S]
    std_ptr,            # *const float32, shape [B*S]
    out_ptr,            # *float32, shape [B*S,F] (output buffer as float32)
    B: tl.constexpr,    # int
    S: tl.constexpr,    # int
    F: tl.constexpr,    # int
    z: tl.constexpr,    # float32, scalar inverse-normal CDF for target_sparsity
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    out_stride_b: tl.constexpr,  # int
    out_stride_s: tl.constexpr,  # int
    out_stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,       # int
):
    pid = tl.program_id(0)  # 0..(B*S-1)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * z

    offs = tl.arange(0, BLOCK_F)
    for start in range(0, F, BLOCK_F):
        idx = start + offs
        mask = idx < F
        x_index = b * stride_b + s * stride_s + idx * stride_f
        x_vals = tl.load(x_ptr + x_index, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        y = tl.maximum(x_vals - threshold, 0.0)
        out_index = b * out_stride_b + s * out_stride_s + idx * out_stride_f
        tl.store(out_ptr + out_index, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        - Computes per-row mean and std along the feature dimension (unbiased=False).
        - Computes inverse-normal CDF z for target_sparsity on host.
        - Applies y = max(input - (mean + std * z), 0) elementwise.
        - Returns output as bfloat16, same shape as inputs.
        """
        if target_sparsity == 0.0:
            return inputs

        assert inputs.is_cuda, "Inputs must be on CUDA for Triton kernels."
        x = inputs.contiguous()
        B, S, F = x.shape
        device = x.device

        # Accumulators
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Heuristic block size for feature dimension
        BLOCK_F = 1024 if F >= 1024 else 256

        # 1) Single-pass reduction: sum and sumsq per row
        grid = (B * S,)
        sum_sumsq_rows_kernel[grid](
            x,
            out_sum,
            out_sumsq,
            B,
            S,
            F,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # 2) Compute mean and std per row
        compute_stats_kernel[grid](
            out_sum,
            out_sumsq,
            out_mean,
            out_std,
            B,
            S,
            F,
        )

        # 3) Compute inverse-normal CDF z on host using erfinv: z = sqrt(2) * erfinv(2p - 1)
        p = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
        z = float(torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv((2.0 * p - 1.0)))

        # 4) Apply threshold in Triton
        out_f32 = torch.empty_like(x, dtype=torch.float32, device=device)

        apply_threshold_kernel[grid](
            x,
            out_mean,
            out_std,
            out_f32,
            B,
            S,
            F,
            z,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out_f32.stride(0),
            out_f32.stride(1),
            out_f32.stride(2),
            BLOCK_F=BLOCK_F,
        )

        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
