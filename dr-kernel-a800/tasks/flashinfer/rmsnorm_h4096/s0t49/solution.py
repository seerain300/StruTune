import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)  # one program per row
    # Pointer to the start of this row
    row_start = row_id * H
    # Accumulator for sum of squares in fp32
    sumsq = 0.0
    # Loop over columns in tiles of BLOCK_SIZE
    # For H=4096 and BLOCK_SIZE=4096 this runs once
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        # Load row segment; Triton will vectorize this
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each element by inv_rms[row] and weight[col]
@triton.jit
def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)  # one program per row
    inv = tl.load(inv_rms_ptr + row_id)  # scalar per row
    # Loop over columns in tiles of BLOCK_SIZE
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = w.to(tl.float32)
        y = x * inv * w
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        # For the provided workload, H is 4096; we specialize for this case.
        assert H == 4096, "This Triton-optimized implementation assumes hidden_size == 4096."

        # Compute in fp32 for numerical stability
        x_fp32 = x.to(torch.float32)  # [B, H]
        w_fp32 = w.to(torch.float32)  # [H]

        # Allocate outputs and per-row factors
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
            w_fp32,
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
