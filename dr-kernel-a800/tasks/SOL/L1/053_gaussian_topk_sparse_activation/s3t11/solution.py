import triton
import triton.language as tl


@triton.jit
def reduce_sum_rows_kernel(x_ptr, sum_ptr, N_rows, L, BLOCK: tl.constexpr):
    """
    Compute per-row sum across the feature dimension. One program per row.
    x_ptr: flattened [N_rows, L]
    sum_ptr: [N_rows], sum for each row
    """
    row = tl.program_id(axis=0)
    acc_sum = 0.0
    start = 0
    while start < L:
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        vals = tl.load(x_ptr + row * L + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(vals, axis=0)
        start += BLOCK
    tl.store(sum_ptr + row, acc_sum)


@triton.jit
def reduce_sumsq_rows_kernel(x_ptr, sumsq_ptr, N_rows, L, BLOCK: tl.constexpr):
    """
    Compute per-row sum of squares across the feature dimension. One program per row.
    """
    row = tl.program_id(axis=0)
    acc_sumsq = 0.0
    start = 0
    while start < L:
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        vals = tl.load(x_ptr + row * L + offs, mask=mask, other=0.0)
        acc_sumsq += tl.sum(vals * vals, axis=0)
        start += BLOCK
    tl.store(sumsq_ptr + row, acc_sumsq)


@triton.jit
def compute_mean_std_threshold_rows_kernel(sum_ptr, sumsq_ptr, threshold_ptr, N_rows, L, multiplier):
    """
    Compute per-row mean, std, and threshold:
      mean[row] = sum[row] / L
      var[row] = sumsq[row] / L - mean[row]^2
      std[row] = sqrt(var[row])
      threshold[row] = mean[row] + std[row] * multiplier
    Write threshold per row (FP32).
    """
    row = tl.program_id(axis=0)
    s = tl.load(sum_ptr + row)
    ss = tl.load(sumsq_ptr + row)
    mean = s / L
    var = ss / L - mean * mean
    std = tl.sqrt(var)
    thr = mean + std * multiplier
    tl.store(threshold_ptr + row, thr)


@triton.jit
def sparse_relu_rows_kernel(x_ptr, threshold_ptr, out_ptr, N_rows, L, BLOCK: tl.constexpr):
    """
    Elementwise sparse ReLU per row with per-row threshold:
      out[row, f] = max(x[row, f] - threshold[row], 0)
    One program per row; iterate over features in chunks of BLOCK.
    """
    row = tl.program_id(axis=0)
    thr_row = tl.load(threshold_ptr + row)
    start = 0
    while start < L:
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x_vals = tl.load(x_ptr + row * L + offs, mask=mask, other=0.0)
        y = x_vals - thr_row
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(out_ptr + row * L + offs, y, mask=mask)
        start += BLOCK


@triton.jit
def cast_bf16_kernel(inp_ptr, out_ptr, N_elems, BLOCK: tl.constexpr):
    """
    Cast FP32 input to BF16 output over N_elems elements (linear indexing).
    Forward MUST invoke this kernel to return BF16 output.
    """
    idx = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_elems
    vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(out_ptr + idx, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version:
        - Compute per-row mean and std along last dim (features).
        - Compute per-row threshold: mean + std * ndtri(target_sparsity).
        - Apply sparse ReLU with per-row thresholds.
        - Cast to bfloat16 via Triton and return.
        """
        assert inputs.dim() == 3, "inputs must be [B, S, L]"
        B, S, L = inputs.shape
        N_rows = B * S
        device = inputs.device

        # Flatten to [N_rows, L] contiguous
        x_flat = inputs.contiguous().view(N_rows, L)

        # Allocate accumulators per row (FP32)
        sum_rows = torch.empty(N_rows, dtype=torch.float32, device=device)
        sumsq_rows = torch.empty(N_rows, dtype=torch.float32, device=device)

        # Launch Triton kernels to compute per-row sum and sumsq across features
        BLOCK = 1024  # features chunk; Triton will specialize per launch
        reduce_sum_rows_kernel[(N_rows,)](x_flat, sum_rows, N_rows, L, BLOCK=BLOCK, num_warps=4)
        reduce_sumsq_rows_kernel[(N_rows,)](x_flat, sumsq_rows, N_rows, L, BLOCK=BLOCK, num_warps=4)

        # Compute per-row threshold (FP32)
        threshold_rows = torch.empty(N_rows, dtype=torch.float32, device=device)
        compute_mean_std_threshold_rows_kernel[(N_rows,)](
            sum_rows, sumsq_rows, threshold_rows, N_rows, L, float(target_sparsity), num_warps=1
        )

        # Elementwise sparse ReLU using per-row threshold (FP32 output)
        out_fp32_flat = torch.empty((N_rows, L), dtype=torch.float32, device=device)
        sparse_relu_rows_kernel[(N_rows,)](x_flat, threshold_rows, out_fp32_flat, N_rows, L, BLOCK=BLOCK, num_warps=4)

        # Cast to bfloat16 via Triton kernel (forward MUST invoke this)
        N_elems = N_rows * L
        out_bf16_flat = torch.empty((N_rows, L), dtype=torch.bfloat16, device=device)
        cast_bf16_kernel[(triton.cdiv(N_elems, 2048),)](out_fp32_flat, out_bf16_flat, N_elems, BLOCK=2048, num_warps=4)

        # Reshape back to [B, S, L]
        out_bf16 = out_bf16_flat.view(B, S, L)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
