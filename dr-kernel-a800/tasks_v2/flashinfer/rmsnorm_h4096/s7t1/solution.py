import torch
import triton
import triton.language as tl

# Kernel 1: compute global sum of squares across all elements of hidden_states (B x H).
# Grid: (B, num_tiles) where num_tiles = ceil_div(H, BLOCK_SIZE)
@triton.jit
def _reduce_sum_squares_kernel(hidden_ptr, sum_ptr,
                                hidden_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(axis=0)  # batch row id
    pid_t = tl.program_id(axis=1)  # tile id along hidden dimension

    row_offset = pid_b * hidden_size
    start = pid_t * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    partial = tl.sum(x_sq, axis=0)  # sum over the vector lane
    # Atomic add into sum_ptr[pid_b]
    tl.atomic_add(sum_ptr + pid_b, partial)


# Kernel 2: scaling + elementwise multiply.
# Grid: (B, num_tiles) where num_tiles = ceil_div(H, BLOCK_SIZE)
@triton.jit
def _scale_and_weight_kernel(hidden_ptr, weight_ptr, out_ptr, inv_rms_global: tl.constexpr,
                             hidden_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)

    row_offset = pid_b * hidden_size
    start = pid_t * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load row slice and weight slice
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Scale row by global inv_rms and multiply by weight
    y = x * inv_rms_global * w
    tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors for Triton kernels
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        # Cast to float32 for stable math (match original behavior)
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        w = weight.contiguous().to(torch.float32)        # [H]

        B, H = x.shape
        # Keep original behavior: hidden_size == 4096
        assert H == 4096, f"hidden_size must be 4096, got {H}"

        # 1) Compute global sum of squares
        sum_squares = torch.zeros(B, device=x.device, dtype=torch.float32)
        BLOCK_SIZE = 1024  # 4096 / 1024 = 4 tiles per row
        grid = (B, triton.cdiv(H, BLOCK_SIZE))
        _reduce_sum_squares_kernel[grid](
            x, sum_squares,
            hidden_size=H, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=2
        )
        total_sum = torch.sum(sum_squares)  # scalar

        # 2) Compute inv_rms_global = rsqrt(mean + EPS), mean over all elements
        mean_global = total_sum / (B * H)
        inv_rms_global = torch.rsqrt(mean_global + self.eps)

        # 3) Scale rows by inv_rms_global and multiply by weight
        out = torch.empty((B, H), device=x.device, dtype=torch.float32)
        grid2 = (B, triton.cdiv(H, BLOCK_SIZE))
        _scale_and_weight_kernel[grid2](
            x, w, out, inv_rms_global.item(),  # pass scalar
            hidden_size=H, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=2
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
