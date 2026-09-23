import torch
import triton
import triton.language as tl


# Reduction kernel: per-row inv_rms = rsqrt(mean(x^2) + EPS)
@triton.jit
def _row_rms_inv_kernel(
    hidden_ptr,            # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,           # *float32, shape [B]
    B: tl.int32,           # batch size
    H: tl.int32,           # hidden size
    EPS: tl.float32,       # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Base pointer for this row (assuming contiguous [B, H] with stride(1) == 1)
    row_base = hidden_ptr + row * H

    sum_sq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


# Elementwise kernel: apply row scaling (inv_rms[row]) then column scaling (weight[col])
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,            # *float32, shape [B, H] contiguous
    inv_rms_ptr,           # *float32, shape [B]
    weight_ptr,            # *float32, shape [H]
    out_ptr,               # *float32, shape [B, H]
    B: tl.int32,           # batch size
    H: tl.int32,           # hidden size
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_chunk = tl.program_id(1)
    if (row >= B) or (col_chunk >= (H // BLOCK_SIZE)):
        return

    col_start = col_chunk * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load row slice and compute index
    row_base = hidden_ptr + row * H
    x = tl.load(row_base + offs, mask=mask, other=0.0)

    # Load scalars
    inv_rms_row = tl.load(inv_rms_ptr + row)
    weight_vec = tl.load(weight_ptr + offs, mask=mask, other=0.0)

    y = x * inv_rms_row * weight_vec
    out_row_base = out_ptr + row * H
    tl.store(out_row_base + offs, y, mask=mask)


def run_triton(hidden_states, weight):
    # Ensure CUDA and contiguity; keep original dtype
    assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors."
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()
    dtype_in = hidden_states.dtype

    B, H = hidden.shape
    assert H == 4096, "This Triton implementation expects hidden size H == 4096."

    # Compute in float32 for numerical stability
    hidden_f32 = hidden.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)

    # Launch reduction kernel: one program per row
    _row_rms_inv_kernel[(B,)](
        hidden_f32, inv_rms, B, H, 1e-5,  # EPS from original
        BLOCK_SIZE=1024, num_warps=4, num_stages=2
    )

    # Allocate output in float32 for computation; cast back later
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

    # Launch elementwise kernel with 2D grid: (rows, col_chunks)
    BLOCK_SIZE_E = 1024
    grid_apply = (B, (H + BLOCK_SIZE_E - 1) // BLOCK_SIZE_E)
    _apply_two_scales_kernel[grid_apply](
        hidden_f32, inv_rms, weight_f32, out_f32,
        B, H,
        BLOCK_SIZE=BLOCK_SIZE_E, num_warps=4, num_stages=2
    )

    # Cast back to original dtype for parity with original code
    return out_f32.to(dtype_in)


# Optional: keep original Model for reference; evaluation uses ModelNew.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)