import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq(
    x_ptr,                # *fp32, shape [B, H]
    inv_rms_ptr,          # *fp32, shape [B]
    B: tl.constexpr,      # batch size (used only for grid)
    H: tl.constexpr,      # hidden size
    EPS,                  # fp32 epsilon
    BLOCK_SIZE: tl.constexpr,  # tile size across hidden dim
):
    row_id = tl.program_id(0)
    # Guard in case grid > B (not necessary if grid=B, but safe)
    if row_id >= B:
        return

    # Accumulate sum of squares for this row in fp32
    sumsq = 0.0
    cols = tl.arange(0, BLOCK_SIZE)
    # Loop over tiles across the hidden dimension
    for start in range(0, H, BLOCK_SIZE):
        offs = start + cols
        mask = offs < H
        x = tl.load(x_ptr + row_id * H + offs, mask=mask, other=0.0)
        # x is fp32
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each element of the row by inv_rms[row] and weight[j]
@triton.jit
def scale_row_elements(
    x_ptr,          # *fp32, input row data [B, H]
    weight_ptr,     # *fp32, weight vector [H]
    inv_rms_ptr,    # *fp32, per-row inv_rms [B]
    out_ptr,        # *fp32, output [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # Load inv_rms for this row
    inv_rms = tl.load(inv_rms_ptr + row_id)  # scalar fp32
    cols = tl.arange(0, BLOCK_SIZE)
    for start in range(0, H, BLOCK_SIZE):
        offs = start + cols
        mask = offs < H
        x = tl.load(x_ptr + row_id * H + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row_id * H + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure we have the expected shape and dtype
        B, H = hidden_states.shape
        # Original code asserts hidden_size == 4096
        assert H == 4096, "This Triton implementation assumes hidden_size == 4096"
        # Ensure contiguity
        x = hidden_states.contiguous().to(torch.float32)  # [B, H] fp32 for compute
        w = weight.contiguous().to(torch.float32)        # [H] fp32

        # Allocate output buffer in fp32
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        # Epsilon as fp32
        EPS = 1e-5

        # Choose kernel config: specialize for H=4096
        BLOCK_SIZE = 4096
        num_warps = 8
        # We previously saw good results with num_stages=3; try that
        num_stages = 3

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq[grid](
            x,
            inv_rms,
            B=B,
            H=H,
            EPS=EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements[grid](
            x,
            w,
            inv_rms,
            out,
            B=B,
            H=H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Cast back to the original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
