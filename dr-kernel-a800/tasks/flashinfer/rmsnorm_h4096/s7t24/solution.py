import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Assumes hidden_size == 4096. Single vector pass for sum of squares, single vector pass for scaling/store.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    row_offset = pid * hidden_size

    # Vector of column indices [0..hidden_size-1]
    offs = tl.arange(0, hidden_size)

    # Pass 1: compute sum of squares across the row
    x = tl.load(hidden_ptr + row_offset + offs)
    x = x.to(tl.float32)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)

    # Compute mean and inv_rms for this row
    mean = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and write output
    w = tl.load(weight_ptr + offs).to(tl.float32)
    y = x * inv_rms * w
    tl.store(out_ptr + row_offset + offs, y)


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

        # Launch fused kernel: one program per row, vectorized across 4096 columns
        _row_fused_scale_kernel[(B,)](
            x, w, out,
            hidden_size=H, EPS=self.eps,
            num_warps=8,   # robust parallelism for 4096-wide vectors
            num_stages=1   # reduce pipeline stages for this simple, memory-bound workload
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
