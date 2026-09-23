import torch
import triton
import triton.language as tl

# Kernel A: compute inv_rms per row
# Grid: (batch_size,)
@triton.jit
def _compute_inv_rms_kernel(hidden_ptr, inv_rms_ptr, hidden_size: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    # Accumulate sum of squares in float32
    sum_sq = 0.0
    # Loop over hidden_size in chunks
    for start in range(0, hidden_size, 1024):
        offs = start + tl.arange(0, 1024)  # vector of indices within the row
        mask = offs < hidden_size
        # Load x[pid, offs], cast to float32
        x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x_sq = x * x
        # Reduce this chunk to scalar and add to accumulator
        # Triton will handle vector reduction; we can sum across the vector
        # using a reduction-like approach by summing elements
        # Note: we need to reduce the vector to scalar. Triton does not have a direct sum,
        # but we can rely on the fact that tl.load returns a vector and arithmetic ops
        # broadcast across the vector, so when we multiply, we can keep a scalar accumulator
        # by explicitly reducing: sum_sq += tl.sum(x_sq, axis=0)
        # However, tl.sum is not available in all versions; instead, we can accumulate
        # elementwise by iterating over the 1024-lane vector and adding to sum_sq.
        # Triton allows per-lane operations; we can do: sum_sq += tl.sum(x_sq, axis=0) if supported.
        # Given potential version differences, we keep it simple: iterate in chunks and sum vector lanes.
        # Since Triton JIT expects loops to be static or known at compile time, we rely on 1024 chunk and vectorized load.
        # Compute the sum of this chunk:
        # sum_sq += tl.sum(x_sq, axis=0)
        sum_sq += tl.sum(x_sq, axis=0)
    # Compute mean and inv_rms
    mean = sum_sq / hidden_size
    inv_r = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + pid, inv_r)


# Kernel B: elementwise scaling per tile (row, cols)
# Grid: (batch_size, ceil_div(hidden_size, BLOCK_SIZE))
@triton.jit
def _scale_and_weight_kernel(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr,
                              hidden_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_row = tl.program_id(axis=0)  # row index
    pid_col = tl.program_id(axis=1)  # tile index along columns
    cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    # Load x[row, cols] and weight[cols] in float32
    x = tl.load(hidden_ptr + pid_row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Load inv_rms for this row
    inv_r = tl.load(inv_rms_ptr + pid_row)  # scalar
    # Compute output: y = x * inv_r * w
    y = x * inv_r * w
    # Store result (out_ptr is assumed to be float32; we will cast on host if needed)
    tl.store(out_ptr + pid_row * hidden_size + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA device; Triton requires CUDA
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        # Cast to float32 for stable math (match original behavior)
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        weight = weight.contiguous().to(torch.float32)    # [H]

        B, H = x.shape
        # Assert hidden size as in original
        assert H == 4096, f"hidden_size must be 4096, got {H}"

        # Output buffer for inv_rms (one per row)
        inv_rms = torch.empty(B, device=x.device, dtype=torch.float32)

        # Launch reduction kernel: compute inv_rms per row
        grid_a = (B,)
        # Choose a reasonable number of warps; 1 program per row, BLOCK_SIZE looped inside
        _compute_inv_rms_kernel[grid_a](
            x, inv_rms, H, self.eps,
            num_warps=4
        )

        # Allocate output tensor in float32 for computation
        out = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # Launch elementwise scaling kernel
        BLOCK_SIZE = 256  # tile size along columns
        grid_b = (B, triton.cdiv(H, BLOCK_SIZE))
        _scale_and_weight_kernel[grid_b](
            x, weight, inv_rms, out,
            H, BLOCK_SIZE,
            num_warps=4
        )

        # Cast back to original dtype to match original behavior
        out = out.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
