import torch
import triton
import triton.language as tl

# Kernel 1: compute row-wise sum of squares in float32, one program per row
@triton.jit
def _row_sums_kernel(hidden_ptr, out_sums_ptr,
                     batch_size, hidden_size: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    sumsq = 0.0
    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        h = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(h * h, axis=0)

    tl.store(out_sums_ptr + pid, sumsq)

# Kernel 2: scale and store y = hidden * rsqrt(sumsq / hidden_size + EPS) * weight, one program per row
@triton.jit
def _scale_store_kernel(hidden_ptr, weight_ptr, out_ptr, out_sums_ptr,
                        batch_size, hidden_size: tl.constexpr, EPS: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    # Load sum of squares for this row (float32)
    sumsq = tl.load(out_sums_ptr + pid)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)  # scalar float32

    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size

        h = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        y = h * inv_rms * w
        tl.store(out_ptr + pid * hidden_size + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device for Triton"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Output buffer in float32 for numerical stability
        out = torch.empty((batch_size, hidden_size), device=hidden_states.device, dtype=torch.float32)

        # Intermediate buffer for row-wise sums
        out_sums = torch.empty((batch_size,), device=hidden_states.device, dtype=torch.float32)

        grid = (batch_size,)
        # Heuristic: choose warps based on batch_size to balance occupancy
        if batch_size < 32:
            num_warps = 8
        elif batch_size < 256:
            num_warps = 16
        else:
            num_warps = 32

        # Kernel 1: compute row-wise sum of squares
        _row_sums_kernel[grid](
            hidden_states, out_sums,
            batch_size, hidden_size,
            BLOCK_SIZE=4096,
            num_warps=num_warps,
            num_stages=1
        )

        # Kernel 2: scale and store
        _scale_store_kernel[grid](
            hidden_states, weight, out, out_sums,
            batch_size, hidden_size, EPS=1e-5,
            BLOCK_SIZE=4096,
            num_warps=num_warps,
            num_stages=1
        )

        # Cast back to the original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
