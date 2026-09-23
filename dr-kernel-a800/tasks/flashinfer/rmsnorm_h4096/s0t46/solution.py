import torch
import triton
import triton.language as tl

# Kernel 1: per-row reduction to compute sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(
    x_ptr,            # *f32, [B, H], contiguous
    inv_rms_ptr,      # *f32, [B]
    B: tl.constexpr,  # batch size (used in grid, not in math)
    H: tl.constexpr,  # hidden size
    EPS,              # f32 scalar
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)  # one program per row
    # If grid might exceed B, guard (defensive)
    if row_id >= B:
        return

    # Accumulate sum of squares in fp32
    sumsq = 0.0
    # Loop over the row in tiles of BLOCK_SIZE
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row_id * H + offs, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each row by per-row inv_rms and weight, write output
@triton.jit
def scale_row_elements_kernel(
    x_ptr,            # *f32, [B, H]
    weight_ptr,       # *f32, [H]
    inv_rms_ptr,      # *f32, [B]
    out_ptr,          # *f32, [B, H]
    B: tl.constexpr,  # batch size (grid size)
    H: tl.constexpr,  # hidden size
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)
    # Loop over the row in tiles of BLOCK_SIZE
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row_id * H + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row_id * H + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and contiguous; compute in fp32
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        B, H = hidden_states.shape
        # For this workload, H is expected to be 4096; we specialize for that case
        assert H == 4096, "This optimized Triton path assumes hidden_size == 4096"

        x = hidden_states.contiguous().to(torch.float32)    # [B, H]
        w = weight.contiguous().to(torch.float32)          # [H]

        # Allocate output (fp32 compute)
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Buffer for per-row inv_rms
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        EPS = 1e-5

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq_kernel[grid](
            x, inv_rms, B, H, EPS,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x, w, inv_rms, out, B, H,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
