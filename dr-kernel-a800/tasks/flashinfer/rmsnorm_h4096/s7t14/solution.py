import torch
import triton
import triton.language as tl

# Fused Triton kernel: each program handles multiple rows (ROWS_PER_PROGRAM)
# and the full hidden_size (BLOCK_SIZE = 4096) in one chunk. Two passes per row:
# pass 1: accumulate sum of squares; pass 2: scale and store.
@triton.jit
def _rowscale_fused_kernel(hidden_ptr, weight_ptr, out_ptr,
                            B, hidden_size: tl.constexpr, EPS: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    pid = tl.program_id(axis=0)  # each program handles a block of rows
    row_start = pid * ROWS_PER_PROGRAM

    # Process up to ROWS_PER_PROGRAM rows in this program
    for r in range(ROWS_PER_PROGRAM):
        row = row_start + r
        row_valid = row < B

        # Accumulator for sum of squares (float32), masked by row_valid
        sum_sq = 0.0

        # Pass 1: compute sum of squares across the row
        for start in range(0, hidden_size, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            col_mask = offs < hidden_size
            row_offset = row * hidden_size
            mask = col_mask & row_valid

            # Load hidden states for this row slice and cast to float32
            x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
            x = x.to(tl.float32)
            x_sq = x * x
            # Reduce this chunk to scalar and add
            sum_sq += tl.sum(x_sq, axis=0)

        # Compute mean and inv_rms for this row
        mean = sum_sq / hidden_size
        inv_rms = tl.rsqrt(mean + EPS)

        # Pass 2: scale and write output
        for start in range(0, hidden_size, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            col_mask = offs < hidden_size
            row_offset = row * hidden_size
            mask = col_mask & row_valid

            # Load x and weight; cast to float32 for math
            x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * inv_rms * w
            tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors for Triton kernels
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        # Make inputs contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        # Keep original behavior: hidden_size == 4096
        assert H == 4096, f"hidden_size must be 4096, got {H}"

        # Output tensor in float32 for computation
        out = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # Choose how many rows per program. 8 is a good balance to reduce grid size
        # without increasing register pressure too much.
        ROWS_PER_PROGRAM = 8
        grid = (triton.cdiv(B, ROWS_PER_PROGRAM),)

        # Launch fused kernel
        _rowscale_fused_kernel[grid](
            x, w, out,
            B, hidden_size=H, EPS=self.eps,
            BLOCK_SIZE=4096, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
            num_warps=8,   # robust parallelism for memory-bound work
            num_stages=1   # reduce pipeline stages for consistency
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
