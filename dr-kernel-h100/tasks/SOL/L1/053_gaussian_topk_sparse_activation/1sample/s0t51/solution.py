import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_row_kernel(x_ptr, mean_ptr, sumsq_ptr, L, K, BLOCK: tl.constexpr):
    """
    One program per row over the flattened dimension L = B*S*K.
    Each program reduces across K for its row, writing mean and sumsq (sum of squares) to output arrays.
    """
    pid = tl.program_id(axis=0)
    # Compute starting index for this row
    # Each row has K elements, so row starts at pid*K
    start = pid * K

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in chunks of BLOCK
    for kk in range(0, K, BLOCK):
        idx = start + kk + tl.arange(0, BLOCK)
        mask = idx < L  # valid elements within the row
        # Load a chunk, masked
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        # Reduce this chunk into scalars
        # Note: tl.sum expects a vector; masked positions are 0
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    # Compute mean and store results
    mean = sum_val / K
    # Store per-row mean and sumsq
    tl.store(mean_ptr + pid, mean)
    tl.store(sumsq_ptr + pid, sumsq_val)


@triton.jit
def _compute_threshold_kernel(mean_ptr, sumsq_ptr, threshold_ptr, L, K, z_score: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per row. Computes std and threshold = mean + std * z_score and stores it.
    """
    pid = tl.program_id(axis=0)
    start = pid * K

    # Load per-row mean and sumsq
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)

    # Compute std = sqrt(sumsq - mean^2) / K
    var = sumsq - mean * mean
    # Clamp to avoid tiny negative due to FP error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)  # divide by K implicit via var definition
    threshold = mean + std * z_score  # z_score is passed as scalar
    tl.store(threshold_ptr + pid, threshold)


@triton.jit
def _apply_sub_relu_kernel(x_ptr, threshold_ptr, out_ptr, L, K, BLOCK: tl.constexpr):
    """
    One program per row. Subtracts threshold from each element and applies ReLU, stores in fp32.
    Assumes out_ptr is a 1D buffer of size L.
    """
    pid = tl.program_id(axis=0)
    start = pid * K

    # Load threshold for this row
    m = tl.load(threshold_ptr + pid)

    for kk in range(0, K, BLOCK):
        idx = start + kk + tl.arange(0, BLOCK)
        mask = idx < L
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        y = vals - m
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block: int = 256, num_warps_reduce: int = 4, num_warps_elem: int = 4):
        super().__init__()
        # Precompute z_score = inverse_normal_cdf(target_sparsity). Using the common value for 0.9.
        # torch.special.ndtri gives more accurate inverse than manual approximation.
        self.z_score = float(torch.special.ndtri(torch.tensor(target_sparsity)))
        self.block = block
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_elem = num_warps_elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous input for simple 1D addressing
        x = x.contiguous()
        # Work in float32 inside Triton
        x_f32 = x.to(torch.float32)

        # Flatten rows: L = B*S*K, one row per program
        B, S, K = x_f32.shape
        L = B * S * K

        # Per-row buffers (fp32 on device)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        threshold_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        out_fp32 = torch.empty(L, dtype=torch.float32, device=x_f32.device)

        # 1) Reduce sum and sumsq per row
        grid = (B * S,)
        _reduce_sum_sumsq_row_kernel[grid](
            x_f32,
            mean_row,
            sumsq_row,
            L,
            K,
            BLOCK=self.block,
            num_warps=self.num_warps_reduce,
            num_stages=2,
        )

        # 2) Compute threshold per row
        _compute_threshold_kernel[grid](
            mean_row,
            sumsq_row,
            threshold_row,
            L,
            K,
            float(self.z_score),  # pass scalar float
            BLOCK=self.block,
            num_warps=self.num_warps_elem,
            num_stages=2,
        )

        # 3) Apply subtraction and ReLU, write to 1D output
        _apply_sub_relu_kernel[grid](
            x_f32,
            threshold_row,
            out_fp32,
            L,
            K,
            BLOCK=self.block,
            num_warps=self.num_warps_elem,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
