import torch
import triton
import triton.language as tl

# Kernel 1: per-row reduction to compute sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    # Guard: if grid > B, skip
    if row_id >= B:
        return

    # Accumulate sum of squares in fp32
    sumsq = 0.0
    # Loop over the row in tiles of size BLOCK_SIZE
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each element using per-row inv_rms and weight
@triton.jit
def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)  # scalar per row
    # For each tile of the row
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # weight is [H]
        x = x.to(tl.float32)
        w = w.to(tl.float32)
        y = x * inv_rms * w  # y[r, j] = x[r, j] * inv_rms[r] * weight[j]
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on the same device and dtype handling
        assert hidden_states.dim() == 2, "hidden_states must be [batch, hidden_size]"
        B, H = hidden_states.shape
        # Compute in fp32 for numerical stability
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        weight_fp32 = weight.contiguous().to(torch.float32)  # [H]
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)
        EPS = 1e-5

        # Choose BLOCK_SIZE based on H
        if H == 4096:
            BLOCK_SIZE = 4096
            num_warps = 8
            num_stages = 3
        else:
            BLOCK_SIZE = 1024
            num_warps = 4
            num_stages = 3

        grid = (B,)

        # Launch reduction kernel
        reduce_row_sumsq_kernel[grid](
            x, inv_rms, B, H, EPS, BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages
        )

        # Launch scaling kernel
        scale_row_elements_kernel[grid](
            x, weight_fp32, inv_rms, out, B, H, BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages
        )

        # Cast back to original dtype of hidden_states for output
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
