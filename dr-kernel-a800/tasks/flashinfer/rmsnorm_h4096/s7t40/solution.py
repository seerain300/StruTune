import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach without unsupported pointer indexing:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms, store per-row inv_rms.
# - Pass 2: reload x and weight, compute y = x * inv_rms[b] * weight, and store to output.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr,
                             batch_size, hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # safety mask; hidden_size == BLOCK_SIZE but keep for generality

    # Load row x and weight
    x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, eviction_policy='evict_last')
    w = tl.load(weight_ptr + offs, mask=mask, eviction_policy='evict_last')

    # Pass 1: sum of squares in float32
    x_fp32 = x.to(tl.float32)
    sumsq = tl.sum(x_fp32 * x_fp32)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)
    # Store per-row inv_rms (fp32) for later use in pass 2
    tl.store(inv_rms_ptr + pid, inv_rms)

    # Pass 2: scale and store
    # Reload x and weight (in case of dtype differences), compute in fp32
    x2 = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, eviction_policy='evict_last')
    w2 = tl.load(weight_ptr + offs, mask=mask, eviction_policy='evict_last')
    x2 = x2.to(tl.float32)
    w2 = w2.to(tl.float32)
    inv_rms_b = tl.load(inv_rms_ptr + pid)  # scalar fp32
    y = (x2 * inv_rms_b) * w2
    # Store output; Triton will cast to out_ptr element type if needed
    tl.store(out_ptr + pid * hidden_size + offs, y, mask=mask)

# Helper to run the Triton kernel and return the result in the original dtype
def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Ensure CUDA tensors
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    # Shapes
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096, "hidden_size must be 4096"

    # Make inputs contiguous
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    # Output in original dtype; kernel will compute in fp32 internally
    out = torch.empty_like(hidden)

    # Per-row inv_rms buffer (fp32)
    inv_rms = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)

    # Launch Triton kernel: one program per row
    grid = (batch_size,)
    _row_fused_scale_kernel[grid](
        hidden, weight, inv_rms, out,
        batch_size=batch_size,
        hidden_size=hidden_size,
        EPS=eps,
        BLOCK_SIZE=hidden_size,
        num_warps=16,
        num_stages=1,
    )

    return out

# Entry point module: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Use Triton kernel; no torch ops inside the kernel
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
