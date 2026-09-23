import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _row_sumsq_kernel(hidden_ptr, sumsq_ptr, B, H, stride_hs, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row sum of squares of hidden states in float32.
    Grid: (B,)
    hidden_ptr: *ptr to [B, H] (row-major), dtype input doesn't matter, cast inside
    sumsq_ptr: FP32 tensor of size B to store per-row sum of squares
    """
    row = tl.program_id(0)
    sumsq = 0.0
    # Loop over columns in chunks of BLOCK_SIZE
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load row elements (dtype inferred from pointer), cast to FP32
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        # Ensure masked lanes don't contribute
        x32 = tl.where(mask, x32, 0.0)
        # Accumulate sum of squares
        sumsq += tl.sum(x32 * x32, axis=0)
    # Store sumsq for this row
    tl.store(sumsq_ptr + row, sumsq)

@triton.jit
def _apply_inv_rms_weight_kernel(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr,
                                 B, H, stride_hs, BLOCK_SIZE: tl.constexpr):
    """
    Apply per-row inv_rms and weight: out[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
    Grid: (B, H) => one program per element (row, col)
    """
    row = tl.program_id(0)
    col = tl.program_id(1)
    # Load x (FP32), weight (FP32), and scalar inv_rms for this row (FP32)
    x = tl.load(hidden_ptr + row * stride_hs + col)  # FP32
    w = tl.load(weight_ptr + col).to(tl.float32)      # FP32
    inv = tl.load(inv_rms_ptr + row)                 # FP32 scalar
    y = x * inv * w
    # Store FP32 output
    tl.store(out_ptr + row * H + col, y)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        """
        Triton-only implementation of run(hidden_states, weight).
        Returns tensor of shape (B, 4096), dtype == hidden_states.dtype.
        """
        assert hidden_states.ndim == 2, "hidden_states must be 2D [B, H]"
        assert hidden_states.shape[1] == HIDDEN_SIZE, "hidden_states second dim must be 4096"
        assert weight.shape == (HIDDEN_SIZE,), "weight must be 1D of length 4096"

        # Ensure CUDA and contiguous
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        device = hidden.device

        # Allocate FP32 buffers for intermediate results
        sumsq = torch.empty(B, dtype=torch.float32, device=device)
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch kernel 1: compute per-row sum of squares in FP32
        grid1 = (B,)
        _row_sumsq_kernel[grid1](
            hidden, sumsq, B, H, hidden.stride(0),
            BLOCK_SIZE=1024,
            num_warps=4,
            num_stages=2,
        )

        # Compute inv_rms per row using torch (simple vector op)
        # inv_rms[row] = rsqrt(sumsq[row] / H + EPS)
        inv_rms = torch.rsqrt(sumsq / float(H) + EPS)

        # Launch kernel 2: apply inv_rms and weight
        grid2 = (B, H)
        _apply_inv_rms_weight_kernel[grid2](
            hidden, weight, inv_rms, out_fp32,
            B, H, hidden.stride(0),
            BLOCK_SIZE=1,  # one element per program
            num_warps=1,
            num_stages=1,
        )

        # Cast to original dtype to match original behavior
        out = out_fp32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
