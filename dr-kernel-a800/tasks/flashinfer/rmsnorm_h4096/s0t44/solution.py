import torch
import triton
import triton.language as tl

# Kernel 1: per-row reduction to compute sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    sumsq = 0.0  # fp32 accumulator

    # Process the row in tiles of size BLOCK_SIZE (here H == BLOCK_SIZE -> single iteration)
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)  # load as original dtype
        x = x.to(tl.float32)  # accumulate in fp32
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)

# Kernel 2: elementwise scaling y = x * inv_rms[row] * weight[j]
@triton.jit
def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    inv_rms = tl.load(inv_rms_ptr + row_id)  # scalar per row, fp32
    factor = inv_rms  # we'll multiply x by this factor and weight

    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)  # load original dtype
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)         # load weight (already fp32 from host)
        x_fp32 = x.to(tl.float32)
        w_fp32 = w.to(tl.float32)  # already fp32; included for clarity
        y = x_fp32 * factor * w_fp32
        tl.store(out_ptr + row_start + cols, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        B, H = hidden_states.shape
        # Specialize for the provided hidden_size
        assert H == 4096, "This implementation specializes for hidden_size == 4096"

        # Ensure contiguity and cast inputs for computation
        x = hidden_states.contiguous()
        weight_fp32 = weight.contiguous().to(torch.float32)  # [H]
        x_fp32 = x.to(torch.float32)  # [B, H], fp32 compute

        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        EPS = 1e-5

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq_kernel[grid](
            x_fp32,
            inv_rms,
            B=B,
            H=H,
            EPS=EPS,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x_fp32,
            weight_fp32,
            inv_rms,
            out,
            B=B,
            H=H,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
