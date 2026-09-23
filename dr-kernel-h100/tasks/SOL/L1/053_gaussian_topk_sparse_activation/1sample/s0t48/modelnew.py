import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_row(x_ptr, mean_ptr, sumsq_ptr,
                           B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                           stride_b, stride_s, stride_k,
                           BLOCK_K: tl.constexpr):
    """
    One program per row (b, s) computes:
      sum = sum(x[b, s, :]), sumsq = sum(x[b, s, :]**2),
      mean_row[b*S + s] = sum / K
      sumsq_row[b*S + s] = sumsq / K
    """
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate across K in tiles
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        vals = tl.load(x_ptr + base + idx * stride_k, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        # Sum and sum of squares for this tile
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Store per-row mean and sumsq (divide by K)
    row_id = b * S + s
    mean_ptr[row_id] = sum_val / K
    sumsq_ptr[row_id] = sumsq_val / K


@triton.jit
def _apply_threshold_relu_row(x_ptr, out_ptr,
                               mean_ptr, sumsq_ptr,
                               B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                               stride_b, stride_s, stride_k,
                               z_score,  # scalar float
                               BLOCK_K: tl.constexpr):
    """
    One program per row (b, s):
      - load mean and sumsq for the row
      - compute std = sqrt(max(sumsq - mean^2, 0))  [population std]
      - m = mean + std * z_score
      - for kk in 0..K-1: out[b, s, kk] = max(0, x[b, s, kk] - m)
    Writes to a 1D out_ptr of length B*S*K in row-major order.
    """
    b = tl.program_id(0)
    s = tl.program_id(1)

    base = b * stride_b + s * stride_s

    # Load mean and sumsq for this row
    row_id = b * S + s
    mean = mean_ptr[row_id]
    sumsq = sumsq_ptr[row_id]

    # Compute population std: var = sumsq - mean^2
    var = sumsq - mean * mean
    # Numerical safety
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Threshold factor
    m = mean + std * z_score

    # Iterate across K in tiles and apply threshold + ReLU
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        vals = tl.load(x_ptr + base + idx * stride_k, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        y = vals - m
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)

        # Compute output linear indices for this row
        out_base = row_id * K
        out_idx = out_base + kk + tl.arange(0, BLOCK_K)
        tl.store(out_ptr + out_idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute z_score once for given sparsity
        # inverse_normal_cdf(p) ≈ norm.ppf(p) ; here p = target_sparsity
        # torch.special.erf is available if present; use norm.ppf fallback
        # We'll use torch's norm.ppf to get accurate value.
        self.z_score = float(torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(target_sparsity)))
        self.block_k = 256  # tile size along K
        self.num_warps_reduce = 4
        self.num_warps_elem = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x of shape [B, S, K]
        # Make input contiguous to simplify strides: for [B,S,K] contiguous, strides are (S*K, K, 1)
        x_contig = x.contiguous().to(torch.float32)
        B, S, K = x_contig.shape
        stride_b, stride_s, stride_k = x_contig.stride()

        # Per-row buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_contig.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_contig.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_row[grid](
            x_contig,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_contig.device)

        _apply_threshold_relu_row[grid](
            x_contig,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,  # scalar float
            self.block_k,
            num_warps=self.num_warps_elem,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)