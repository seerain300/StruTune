import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _row_sumsq_kernel(hidden_ptr, sumsq_ptr, B, stride_hs, BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Computes sum of squares of that row in FP32.
    hidden_ptr: *float32 (input is cast to float32 in host before launch)
    sumsq_ptr: *float32 (per-row sum of squares)
    B: number of rows
    stride_hs: row stride in elements (for 2D tensor, typically HIDDEN_SIZE)
    BLOCK_SIZE: chunk size for vectorized loads
    """
    row = tl.program_id(axis=0)
    # Initialize sum of squares
    sumsq = 0.0
    # Iterate over the row in chunks
    for col in range(0, HIDDEN_SIZE, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < HIDDEN_SIZE
        # Load hidden[row, offs] (float32), masked
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        # Zero out masked lanes to avoid contributing
        x = tl.where(mask, x, 0.0)
        # Accumulate sum of squares
        sumsq += tl.sum(x * x, axis=0)
    # Store per-row sum of squares
    tl.store(sumsq_ptr + row, sumsq)


@triton.jit
def _apply_inv_rms_and_weight_kernel(hidden_fp32_ptr, weight_fp32_ptr, inv_rms_ptr, out_fp32_ptr,
                                     B, H, stride_hs, BLOCK_SIZE: tl.constexpr):
    """
    One program per (row, col) element. Computes y[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
    All inputs are float32. Output written as float32.
    grid: (B, H)
    """
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    # Load hidden[row, col] as float32
    x = tl.load(hidden_fp32_ptr + row * stride_hs + col)
    # Load inv_rms[row] and weight[col]
    inv_rms = tl.load(inv_rms_ptr + row)
    w = tl.load(weight_fp32_ptr + col)
    # Compute y
    y = x * inv_rms * w
    # Store output
    tl.store(out_fp32_ptr + row * stride_hs + col, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation:
          - hidden_states: [B, 4096], bfloat16 (converted to float32 for computation)
          - weight: [4096], bfloat16 (converted to float32 for computation)
          - Output: [B, 4096], cast back to hidden_states.dtype
        """
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == HIDDEN_SIZE, "hidden_states must be [B, 4096]"
        assert weight.dim() == 1 and weight.shape[0] == HIDDEN_SIZE, "weight must be [4096]"

        B, H = hidden_states.shape
        device = hidden_states.device

        # Ensure contiguous
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Allocate buffers
        sumsq = torch.empty((B,), dtype=torch.float32, device=device)

        # Launch kernel A: compute sum of squares per row
        _row_sumsq_kernel[(B,)](hidden, sumsq, B, H, HIDDEN_SIZE, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Compute inv_rms per row on device using torch (minimal and necessary)
        inv_rms = torch.rsqrt(sumsq / float(H) + EPS)  # FP32

        # Prepare float32 versions of inputs for kernel B
        hidden_fp32 = hidden.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # Output buffer (FP32), we will cast to original dtype after kernel
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch kernel B: apply inv_rms and weight
        _apply_inv_rms_and_weight_kernel[(B, H)](hidden_fp32, weight_fp32, inv_rms, out_fp32, B, H, HIDDEN_SIZE, BLOCK_SIZE=1, num_warps=2, num_stages=2)

        # Match original behavior: return in hidden_states.dtype
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
