import torch
import triton
import triton.language as tl

# Triton kernels: reduction and scaling
# Kernel 1: per-row sum of squares -> inv_rms
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over columns in tiles of size BLOCK_SIZE
    for start in range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_off = row_id * H + cols
        x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)

# Kernel 2: scale row elements using per-row inv_rms and weight
@triton.jit
def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)  # scalar per row

    for start in range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_off = row_id * H + cols
        x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals
        tl.store(out_ptr + x_off, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # hidden_states: [B, H], weight: [H], dtype bfloat16 as per get_inputs
        B, H = hidden_states.shape

        # Ensure contiguous tensors and compute in fp32
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        weight_fp32 = weight.contiguous().to(torch.float32)  # [H]

        # Allocate output buffer and per-row inv_rms
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        EPS = 1e-5

        # Fast path specialized for H == 4096
        if H == 4096:
            grid = (B,)
            reduce_row_sumsq_kernel[grid](
                x,
                inv_rms,
                B=B,
                H=H,
                EPS=EPS,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=3,
            )

            scale_row_elements_kernel[grid](
                x,
                weight_fp32,
                inv_rms,
                out,
                B=B,
                H=H,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=3,
            )
        else:
            # Generic fallback for other H sizes
            BLOCK_SIZE = 1024
            grid = (B,)
            reduce_row_sumsq_kernel[grid](
                x,
                inv_rms,
                B=B,
                H=H,
                EPS=EPS,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=3,
            )

            scale_row_elements_kernel[grid](
                x,
                weight_fp32,
                inv_rms,
                out,
                B=B,
                H=H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=3,
            )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
