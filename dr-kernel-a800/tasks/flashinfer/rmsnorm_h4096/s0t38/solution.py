import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
# One Triton program handles one row. It loops over the hidden dimension in tiles of BLOCK_SIZE.
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, out_inv_ptr, H: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    sumsq = 0.0
    # Static loop over columns in tiles of BLOCK_SIZE
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row * H + cols
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)
    inv_rms = tl.rsqrt(sumsq / H + EPS)
    tl.store(out_inv_ptr + row, inv_rms)


# Kernel 2: scale row elements using inv_rms[row] and weight
@triton.jit
def scale_row_elements_kernel(x_ptr, inv_ptr, weight_ptr, out_ptr, H: tl.constexpr,
                               BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    inv = tl.load(inv_ptr + row)  # scalar per row
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row * H + cols
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y_vals = x_vals * inv * w_vals
        out_row_ptr = out_ptr + row * H + cols
        tl.store(out_row_ptr, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Compute in fp32
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        B, H = x_fp32.shape
        device = x_fp32.device

        # Output buffer in fp32 and per-row inv_rms
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=device)

        # Choose BLOCK_SIZE and tuning params
        # Specialize for H == 4096
        if H == 4096:
            BLOCK_SIZE = 4096
            num_warps = 8
            num_stages = 4
        else:
            BLOCK_SIZE = 1024
            num_warps = 4
            num_stages = 3

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq_kernel[grid](
            x_fp32, inv_rms, H, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x_fp32, inv_rms, w_fp32, out_fp32, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages
        )

        # Cast back to original dtype
        out = out_fp32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
