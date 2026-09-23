import torch
import triton
import triton.language as tl

# Row-wise reduction kernel: compute sum of squares per row
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, out_ptr, H, EPS, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    # Accumulate sum of squares in fp32
    sumsq = 0.0
    offs = 0
    while offs < H:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(x_ptr + row * H + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(out_ptr + row, inv_rms)

# Precompute per-element scale: scale = inv_rms[row] * weight[j] for a single row
@triton.jit
def precompute_scale_row_kernel(inv_ptr, weight_ptr, scale_ptr, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    inv = tl.load(inv_ptr + row)  # fp32 scalar
    offs = 0
    while offs < H:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        s = inv * w  # fp32 scale per column
        tl.store(scale_ptr + row * H + idx, s, mask=mask)
        offs += BLOCK_SIZE

# Scaling kernel: y[row, j] = x[row, j] * scale[row, j]
@triton.jit
def scale_row_elements_kernel(x_ptr, scale_ptr, y_ptr, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = 0
    while offs < H:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(x_ptr + row * H + idx, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + row * H + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * s
        # Store as fp32; host will cast to original dtype after kernel
        tl.store(y_ptr + row * H + idx, y, mask=mask)
        offs += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure contiguity
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        # Compute in fp32 inside Triton
        x32 = x.to(torch.float32)
        w32 = w.to(torch.float32)

        # Allocate outputs
        inv_rms = torch.empty((B,), device=x.device, dtype=torch.float32)
        # Use specialized BLOCK_SIZE for H == 4096, else masked kernel
        if H == 4096:
            BLOCK = 4096
            num_warps = 8
            num_stages = 4
        else:
            BLOCK = 1024
            num_warps = 4
            num_stages = 3

        # Launch reduction kernel: one program per row
        grid_reduce = (B,)
        reduce_row_sumsq_kernel[grid_reduce](x32, inv_rms, H, 1e-5, BLOCK_SIZE=BLOCK, num_warps=num_warps, num_stages=num_stages)

        # Precompute per-element scale = inv_rms[row] * weight[j]
        scale = torch.empty((B, H), device=x.device, dtype=torch.float32)
        grid_scale = (B,)
        precompute_scale_row_kernel[grid_scale](inv_rms, w32, scale, H, BLOCK_SIZE=BLOCK, num_warps=num_warps, num_stages=num_stages)

        # Allocate final output (fp32 for compute, cast after)
        y32 = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # Launch scaling kernel: one program per row
        grid_scale2 = (B,)
        scale_row_elements_kernel[grid_scale2](x32, scale, y32, H, BLOCK_SIZE=BLOCK, num_warps=num_warps, num_stages=num_stages)

        # Cast back to original dtype for return
        return y32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
