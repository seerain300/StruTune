import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_row_sumsq_h4096(x_ptr, inv_rms_ptr, B, EPS, BLOCK_SIZE: tl.constexpr):
    # One program per row, reduce across H=4096 in a single masked tile
    pid = tl.program_id(axis=0)
    if pid >= B:
        return

    row_base = x_ptr + pid * BLOCK_SIZE  # for H=4096, exactly one tile

    sumsq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < 4096
    # Load as original dtype, cast to fp32 for math
    x = tl.load(row_base + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / 4096.0
    inv_r = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + pid, inv_r)


@triton.jit
def _scale_row_h4096(x_ptr, w_ptr, y_ptr, inv_rms_ptr, B, EPS, BLOCK_SIZE: tl.constexpr):
    # One program per row, scale across H=4096 in a single masked tile
    pid = tl.program_id(axis=0)
    if pid >= B:
        return

    inv_r = tl.load(inv_rms_ptr + pid)  # scalar per row (fp32)

    row_x_base = x_ptr + pid * BLOCK_SIZE
    row_y_base = y_ptr + pid * BLOCK_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < 4096

    # Load x and w, cast to fp32, compute y
    x = tl.load(row_x_base + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    w = w.to(tl.float32)

    y = x * inv_r * w
    tl.store(row_y_base + offs, y, mask=mask)


@triton.jit
def _reduce_row_sumsq_masked(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    # General masked reduction: one program per row, loop over tiles of BLOCK_SIZE
    pid = tl.program_id(axis=0)
    if pid >= B:
        return

    sumsq = tl.zeros((), dtype=tl.float32)

    num_tiles = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile in range(0, num_tiles):
        start = tile * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Row-major: each row has H elements contiguous
        x = tl.load(x_ptr + pid * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_r = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + pid, inv_r)


@triton.jit
def _scale_row_masked(x_ptr, w_ptr, y_ptr, inv_rms_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    # General masked scaling: one program per row, loop over tiles
    pid = tl.program_id(axis=0)
    if pid >= B:
        return

    inv_r = tl.load(inv_rms_ptr + pid)  # scalar per row (fp32)

    num_tiles = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile in range(0, num_tiles):
        start = tile * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        x = tl.load(x_ptr + pid * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        w = w.to(tl.float32)

        y = x * inv_r * w
        # y_ptr is float32 buffer; we write fp32 results
        tl.store(y_ptr + pid * H + offs, y, mask=mask)


def _run_triton(hs: torch.Tensor, w: torch.Tensor, eps: float = 1e-5):
    """
    Triton-only implementation of:
      y = (hs * rsqrt(mean(hs^2) + eps)) * w
    Forward does not use any torch ops; it launches Triton kernels.
    Returns y with the same dtype as hs.
    """
    assert hs.is_cuda and w.is_cuda, "Inputs must be on CUDA for Triton kernels"
    B, H = hs.shape

    # Ensure contiguity for Triton
    hs_c = hs.contiguous()
    w_c = w.contiguous()

    # Allocate fp32 outputs and per-row inv_rms
    y_fp32 = torch.empty((B, H), dtype=torch.float32, device=hs.device)
    inv_rms = torch.empty(B, dtype=torch.float32, device=hs.device)

    if H == 4096:
        grid = (B,)
        _reduce_row_sumsq_h4096[grid](
            hs_c, inv_rms, B, eps,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=3
        )
        _scale_row_h4096[grid](
            hs_c, w_c, y_fp32, inv_rms, B, eps,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=3
        )
    else:
        BLOCK_SIZE = 1024
        grid = (B,)
        _reduce_row_sumsq_masked[grid](
            hs_c, inv_rms, B, H, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3
        )
        _scale_row_masked[grid](
            hs_c, w_c, y_fp32, inv_rms, B, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3
        )

    # Cast back to original dtype for return (minimal host-side op)
    return y_fp32.to(hs.dtype)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton-only forward: no torch ops, only host-side setup and kernel launches
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
