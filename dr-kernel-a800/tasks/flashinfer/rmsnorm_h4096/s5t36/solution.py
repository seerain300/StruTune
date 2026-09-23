import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _row_normalize_scale_kernel(hidden_ptr, weight_ptr, out_ptr, inv_rms_ptr,
                                 B, H, stride_hs, stride_w, stride_out, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel:
    - Grid: one program per row (grid=(B,))
    - Two passes per row:
      1) Accumulate sum of squares of hidden[row, :] in FP32.
         Compute inv_rms = rsqrt(sumsq / H + EPS) and store per row.
      2) Compute y = hidden_fp32 * inv_rms[row] * weight[col] and store to out (FP32).
    """
    row = tl.program_id(axis=0)

    # Pass 1: sum of squares across the row
    sumsq = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        x32 = tl.where(mask, x32, 0.0)  # ensure masked lanes don't contribute
        sumsq += tl.sum(x32 * x32, axis=0)

    inv_rms = tl.rsqrt(sumsq / H + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)

    # Pass 2: compute output
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_rms_ptr + row)  # scalar for this row
        y = x * inv * w
        tl.store(out_ptr + row * stride_out + offs, y, mask=mask)

def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure CUDA tensors and contiguous
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()
    B, H = hidden.shape
    assert H == HIDDEN_SIZE, "Hidden size must be 4096."
    assert weight.numel() == H, "Weight size must match hidden size."

    # Allocate FP32 output and per-row inv_rms buffer
    out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
    inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)

    # Launch Triton kernel: one program per row
    grid = (B,)
    _row_normalize_scale_kernel[grid](
        hidden, weight, out_fp32, inv_rms,
        B, H, hidden.stride(0), weight.stride(0), out_fp32.stride(0),
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=2
    )

    # Cast back to original dtype to match original behavior
    return out_fp32.to(hidden_states.dtype)

# Entry point model required by evaluation
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return run(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
