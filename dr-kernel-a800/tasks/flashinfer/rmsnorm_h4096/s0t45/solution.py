import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(
    x_ptr,                      # *f32, [B, H]
    inv_rms_ptr,                # *f32, [B]
    B: tl.constexpr,            # batch size (runtime value, Triton will handle)
    H: tl.constexpr,            # hidden size (runtime value)
    EPS: tl.constexpr,          # epsilon
    BLOCK_SIZE: tl.constexpr,   # tile size along hidden dimension
):
    row_id = tl.program_id(0)
    # guard in case grid > B (not needed if grid=B, but safe)
    if row_id >= B:
        return

    # Accumulator for sum of squares
    sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x_row_ptr = x_ptr + row_id * H + idx
        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 2: scale each element by inv_rms[row] and weight[j], write to output
@triton.jit
def scale_row_elements_kernel(
    x_ptr,                   # *f32, [B, H]
    weight_ptr,              # *f32, [H]
    inv_rms_ptr,             # *f32, [B]
    out_ptr,                 # *f32, [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)
    # Iterate over hidden dimension tiles
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x_row_ptr = x_ptr + row_id * H + idx
        w_ptr = weight_ptr + idx
        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        w = tl.load(w_ptr, mask=mask, other=0.0)
        y = x * inv_rms * w
        out_row_ptr = out_ptr + row_id * H + idx
        tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Cast to fp32 for computation
        x_f32 = x.to(torch.float32)
        w_f32 = w.to(torch.float32)

        # Shapes
        B, H = x_f32.shape

        # Allocate output in fp32
        out = torch.empty((B, H), dtype=torch.float32, device=x_f32.device)

        # Per-row inv_rms buffer
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x_f32.device)

        EPS = 1e-5

        # Choose launch parameters based on H
        if H == 4096:
            BLOCK_SIZE = 4096
            num_warps = 8
            num_stages = 4
        else:
            # General fallback
            BLOCK_SIZE = 1024
            num_warps = 4
            num_stages = 3

        # Grid: one program per row
        grid = (B,)

        # Launch reduction kernel
        reduce_row_sumsq_kernel[grid](
            x_f32,
            inv_rms,
            B=B,
            H=H,
            EPS=EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Launch scaling kernel
        scale_row_elements_kernel[grid](
            x_f32,
            w_f32,
            inv_rms,
            out,
            B=B,
            H=H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Cast output back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
