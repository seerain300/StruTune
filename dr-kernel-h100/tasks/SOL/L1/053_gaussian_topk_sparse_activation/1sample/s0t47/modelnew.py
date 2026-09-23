import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_row(
    x_ptr,          # *float32, input tensor pointer
    mean_ptr,       # *float32, output mean per (b, s) row
    sumsq_ptr,      # *float32, output sumsq per (b, s) row
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # seq_len
    K: tl.constexpr,   # intermediate_size
    stride_b,       # stride along batch
    stride_s,       # stride along seq_len
    stride_k,       # stride along feature dim (last)
    BLOCK_K: tl.constexpr,
):
    # program id maps to (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulators
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in tiles
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        offs = base + idx * stride_k
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # accumulate in fp32
        vals = vals.to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Compute mean and sum of squares per row
    mean = sum_val / K
    sumsq = sumsq_val / K  # population sum of squares
    # write results
    row_id = b * S + s
    tl.store(mean_ptr + row_id, mean)
    tl.store(sumsq_ptr + row_id, sumsq)


@triton.jit
def _apply_threshold_relu_row(
    x_ptr,          # *float32, input tensor pointer
    out_ptr,        # *float32, output 1D buffer pointer
    mean_ptr,       # *float32, mean per (b, s) row
    sumsq_ptr,      # *float32, sumsq per (b, s) row
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # seq_len
    K: tl.constexpr,   # intermediate_size
    stride_b,       # stride along batch
    stride_s,       # stride along seq_len
    stride_k,       # stride along feature dim (last)
    z_score,        # scalar float (inverse CDF of target sparsity)
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    base = b * stride_b + s * stride_s
    row_id = b * S + s

    # Load per-row mean and sumsq
    mean = tl.load(mean_ptr + row_id)
    sumsq = tl.load(sumsq_ptr + row_id)

    # Compute std
    var = max(sumsq - mean * mean, 0.0)
    std = tl.sqrt(var)
    threshold = mean + std * z_score

    # Apply ReLU(x - threshold) over K
    for kk in range(0, K, BLOCK_K):
        idx = kk + tl.arange(0, BLOCK_K)
        mask = idx < K
        offs = base + idx * stride_k
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = vals - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        out_offs = row_id * K + kk + tl.arange(0, BLOCK_K)
        tl.store(out_ptr + out_offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute z_score once for target_sparsity
        # For sparsity p, z = inverse_normal_cdf(p). Using torch.special.ndtri on CPU.
        import torch
        self.z_score = float(torch.special.ndtri(torch.tensor(target_sparsity)))
        # Tunable tile size along K
        self.block_k = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, K]
        B, S, K = x.shape
        # Compute in fp32 for numerical stability, ensure contiguous
        x_f32 = x.to(torch.float32).contiguous()
        stride_b, stride_s, stride_k = x_f32.stride()

        # Per-row buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_row[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_f32.device)

        _apply_threshold_relu_row[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,  # scalar float
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)