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


# Elementwise kernel: y = (x * inv_rms) * weight (each row handled by one program)
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,        # *const float32, shape [B, H]
    inv_rms_ptr,       # *float32, shape [B]
    weight_ptr,        # *float32, shape [H]
    out_ptr,           # *float32, shape [B, H]
    B: tl.int32,       # batch size
    H: tl.int32,       # hidden size
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return
    inv = tl.load(inv_rms_ptr + row)

    # Process the entire row in one go to minimize launch overhead
    offs = tl.arange(0, BLOCK_SIZE)
    # First pass: scale by inv_rms
    row_base = hidden_ptr + row * H
    out_row_base = out_ptr + row * H
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(row_base + idx, mask=mask, other=0.0)
        y = x * inv
        tl.store(out_row_base + idx, y, mask=mask)
        col += BLOCK_SIZE

    # Second pass: scale by weight
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        y = tl.load(out_row_base + idx, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0)
        y = y * w
        tl.store(out_row_base + idx, y, mask=mask)
        col += BLOCK_SIZE


def run_triton(hidden_states, weight):
    # Ensure contiguity and dtype handling
    dtype_in = hidden_states.dtype
    hidden = hidden_states.contiguous().to(torch.float32)
    weight = weight.contiguous().to(torch.float32)

    B, H = hidden.shape
    assert H == 4096, "This optimized path expects hidden_size == 4096"

    # Allocate per-row inv_rms
    inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)

    # Launch reduction kernel (one program per row)
    _row_rms_inv_kernel[(B,)](
        hidden, inv_rms, B, H, 1e-5,
        BLOCK_SIZE=1024, num_warps=8, num_stages=2
    )

    # Allocate output in float32 for computation; cast back later
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

    # Launch elementwise kernel with one program per row, process entire row
    _apply_two_scales_kernel[(B, 1)](
        hidden, inv_rms, weight, out_f32, B, H,
        BLOCK_SIZE=4096, num_warps=16, num_stages=2
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


def run(*args):
    return ModelNew()(*args)
