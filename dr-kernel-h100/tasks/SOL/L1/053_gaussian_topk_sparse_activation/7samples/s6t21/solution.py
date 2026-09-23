import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_sparsify_relu_kernel(X, B, S, F, icdf, OUT, BLOCK: tl.constexpr):
    """
    For each row (b, s), compute mean and population std across last dim (F),
    then apply sparsification: y = max(0, x - (mean + std * icdf)).
    Inputs/Outputs are float32.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: compute mean and sum of squares
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x_ptrs = X + b * S * F + s * F + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        # Reduce within the vector
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    # population std (unbiased=False)
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Compute cutoff threshold per row: mean + std * icdf
    cutoff = mean + std * icdf

    # Second pass: sparsify and write output
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x_ptrs = X + b * S * F + s * F + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        out_ptrs = OUT + b * S * F + s * F + idx
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        - Compute per-(batch, seq) row mean and std across features (last dim).
        - Compute icdf for target_sparsity using A&S 5.2.23 central region.
        - Apply ReLU on (inputs - (mean + std * icdf)).

        Returns:
        Tensor of shape [batch_size, seq_len, intermediate_size] in bfloat16.
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            # Return original dtype (bfloat16)
            return inputs

        # Ensure CUDA and contiguous
        x = inputs
        assert x.is_cuda, "ModelNew requires CUDA tensors"
        x = x.contiguous()
        B, S, F = x.shape

        # Compute icdf for target_sparsity (scalar) using Triton with A&S central region
        # We pass a 1-element tensor for icdf; Triton will compute into it.
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        # Use BLOCK=1 for the scalar kernel; num_warps=1
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Allocate output in fp32 for compute, later cast to bfloat16
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Choose BLOCK and warps; keep robust constants
        if F >= 2048:
            BLOCK = 2048
            num_warps = 8
        else:
            BLOCK = 1024
            num_warps = 4

        # Launch one program per (b, s) row
        grid = (B * S,)
        _rowwise_sparsify_relu_kernel[grid](
            x, B, S, F, icdf, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
