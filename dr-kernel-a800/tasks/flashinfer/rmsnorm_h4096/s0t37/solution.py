import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
# One Triton program handles one row. It loops over the hidden dimension in tiles.
@triton.jit
def reduce_row_sumsq_kernel(
    x_ptr,          # *const float (input hidden states, float32)
    out_ptr,        # *float (output per-row inv_rms, float32)
    B: tl.constexpr,  # batch size (not used directly, but kept for clarity)
    H: tl.constexpr,  # hidden size
    EPS: tl.constexpr,  # epsilon
    BLOCK_SIZE: tl.constexpr  # tile size for the hidden dimension
):
    row = tl.program_id(0)  # one program per row
    # Accumulator for sum of squares in fp32
    sumsq = tl.zeros((), dtype=tl.float32)
    # Loop over hidden dimension in tiles
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load row tile (contiguous across columns)
        x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0)
        x2 = x * x
        # Reduce tile to scalar and accumulate
        sumsq += tl.sum(x2, axis=0)
    # Compute inv_rms: rsqrt(mean + EPS) where mean = sumsq / H
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    # Store per-row result
    tl.store(out_ptr + row, inv_rms)


# Kernel 2: scale rows using inv_rms and weight
# One Triton program handles one row. It writes the output y in float32.
@triton.jit
def scale_row_elements_kernel(
    x_ptr,          # *const float (input hidden states, float32)
    weight_ptr,     # *const float (weight, float32)
    inv_ptr,        # *const float (per-row inv_rms, float32)
    y_ptr,          # *float (output, float32)
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # hidden size
    BLOCK_SIZE: tl.constexpr  # tile size
):
    row = tl.program_id(0)
    inv = tl.load(inv_ptr + row)  # scalar per row
    # Loop over hidden dimension in tiles
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(y_ptr + row * H + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous and in float32 for compute
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        w = weight.contiguous().to(torch.float32)        # [H]

        B, H = x.shape
        EPS = 1e-5

        # Output buffer in float32 (compute dtype)
        y = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # Allocate per-row inv_rms
        inv_rms = torch.empty((B,), device=x.device, dtype=torch.float32)

        # Choose BLOCK_SIZE based on H for robustness
        if H == 4096:
            BLOCK_SIZE = 4096
            grid = (B,)
            # Tuned launch parameters for H=4096
            reduce_row_sumsq_kernel[grid](
                x, inv_rms, B, H, EPS, BLOCK_SIZE,
                num_warps=8, num_stages=4
            )
            scale_row_elements_kernel[grid](
                x, w, inv_rms, y, B, H, BLOCK_SIZE,
                num_warps=8, num_stages=4
            )
        else:
            BLOCK_SIZE = 1024
            grid = (B,)
            reduce_row_sumsq_kernel[grid](
                x, inv_rms, B, H, EPS, BLOCK_SIZE,
                num_warps=4, num_stages=3
            )
            scale_row_elements_kernel[grid](
                x, w, inv_rms, y, B, H, BLOCK_SIZE,
                num_warps=4, num_stages=3
            )

        # Return in the original dtype
        return y.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
