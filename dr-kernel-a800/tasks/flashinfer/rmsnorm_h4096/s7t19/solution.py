import torch
import triton
import triton.language as tl

# Fused Triton kernel: one program per row.
# Pass 1: compute sum of squares across the row (single-tile since hidden_size == BLOCK_SIZE).
# Pass 2: scale and write output using the computed inv_rms.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    row_offset = pid * hidden_size

    # Load and compute sum of squares for the entire row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)  # scalar

    # Compute inv_rms for this row
    mean = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and write output
    x2 = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x2 * inv_rms * w
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

        # Launch fused kernel: one program per row
        # Use full row tile to minimize loop iterations
        BLOCK_SIZE = 4096
        _row_fused_scale_kernel[(B,)](
            x, w, out,
            hidden_size=H, EPS=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,  # robust parallelism for memory-bound work
            num_stages=2   # allow some pipelining of loads/stores
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
