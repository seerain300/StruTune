import torch
import triton
import triton.language as tl

# Kernel 1: per-row reduction to compute sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq(
    x_ptr,          # *float32, [B, H]
    inv_rms_ptr,    # *float32, [B]
    B, H,           # int32
    EPS,            # float32
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    row_base = x_ptr + row_id * H

    # Accumulate sum of squares across the row in fp32
    sumsq = 0.0
    offs = 0
    while offs < H:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(row_base + idx, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    # Compute inv_rms = rsqrt(mean + EPS)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each element by inv_rms[row] and weight[j]
@triton.jit
def scale_row_elements(
    x_ptr,            # *float32, [B, H]
    weight_ptr,       # *float32, [H]
    inv_rms_ptr,      # *float32, [B]
    out_ptr,          # *float32, [B, H]
    B, H,             # int32
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)

    row_x = x_ptr + row_id * H
    row_out = out_ptr + row_id * H

    offs = 0
    while offs < H:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(row_x + idx, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(row_out + idx, y, mask=mask)
        offs += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape

        # Optimized path assumes hidden_size == 4096 (matches provided get_inputs)
        assert H == 4096, "This optimized Triton path assumes hidden_size == 4096"

        # Compute in fp32 for numerical stability
        x = hidden_states.to(torch.float32)           # [B, H]
        weight_fp32 = weight.to(torch.float32)        # [H]

        # Allocate output buffer
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        # Buffer for per-row inv_rms
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        EPS = 1e-5

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq[grid](
            x,
            inv_rms,
            B=B, H=H, EPS=EPS,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=4,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements[grid](
            x,
            weight_fp32,
            inv_rms,
            out,
            B=B, H=H,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=4,
        )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
