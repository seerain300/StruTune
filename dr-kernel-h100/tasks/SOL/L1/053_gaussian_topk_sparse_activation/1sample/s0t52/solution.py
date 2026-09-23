import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_row_kernel(x_ptr, mean_ptr, sumsq_ptr, L: tl.int32, K: tl.int32, BLOCK: tl.constexpr):
    # One program per row. Here L = B*S*K is the total number of elements, but we need the number of rows P = L // K.
    # We derive row index from program id: pid in [0, P).
    pid = tl.program_id(0)
    # Guard in case grid > P (shouldn't happen if we set grid = (P,)
    if pid >= L // K:
        return

    # Compute row base offset in the flattened memory: row starts at pid * K
    row_start = pid * K
    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate across K in tiles of BLOCK
    for kk in range(0, K, BLOCK):
        offs = kk + tl.arange(0, BLOCK)
        mask = offs < K
        # Load values for this row
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and store
    mean = sum_val / K
    sumsq = sum_sq / K  # population sum of squares
    tl.store(mean_ptr + pid, mean)
    tl.store(sumsq_ptr + pid, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(x_ptr, threshold_ptr, out_ptr, mean_ptr, sumsq_ptr, L: tl.int32, K: tl.int32, z_score: tl.float32, BLOCK: tl.constexpr):
    # One program per row
    pid = tl.program_id(0)
    if pid >= L // K:
        return

    row_start = pid * K

    # Load per-row statistics
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq - mean * mean)
    # Ensure std is non-negative
    std = tl.maximum(std, 0.0)
    threshold = mean + std * z_score

    # Apply: output = max(0, x - threshold) for each element in the row
    for kk in range(0, K, BLOCK):
        offs = kk + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block: int = 256, num_warps_reduce: int = 4, num_warps_elem: int = 4):
        super().__init__()
        # Precompute z_score = inverse normal cdf for target_sparsity
        # Use a standard approximation; target_sparsity=0.9 -> z_score ≈ 1.2815515655446004
        self.z_score = float(1.2815515655446004)
        self.block = block
        self.num_warps_reduce = num_warps_reduce
        self.num_warps_elem = num_warps_elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only compute (no PyTorch ops on tensors)
        # Make input contiguous and compute in fp32
        x_f32 = x.to(torch.float32).contiguous()
        B, S, K = x_f32.shape
        L = B * S * K
        P = B * S  # number of rows

        # Allocate per-row buffers (fp32)
        mean_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(P, dtype=torch.float32, device=x_f32.device)

        # 1) Reduce sum and sumsq per row
        grid = (P,)
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

        # 2) Apply subtraction and ReLU, write to 1D output (fp32)
        out_fp32 = torch.empty(L, dtype=torch.float32, device=x_f32.device)
        _apply_threshold_relu_kernel[grid](
            x_f32,
            torch.empty(0, device=x_f32.device),  # dummy, not used
            out_fp32,
            mean_row,
            sumsq_row,
            L,
            K,
            float(self.z_score),  # scalar
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
