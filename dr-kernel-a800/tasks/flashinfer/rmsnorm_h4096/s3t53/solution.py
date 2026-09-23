import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel for H == 4096: process one full row per program.
@triton.jit
def fullrow_scale_kernel(
    hidden_ptr,       # *float32, shape [B, H]
    weight_ptr,       # *float32, shape [H]
    out_ptr,          # *float32, shape [B, H]
    B: tl.constexpr,  # number of rows (batch size)
    H: tl.constexpr,  # hidden size
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Column indices for the full row
    offs = tl.arange(0, BLOCK_SIZE)
    # Load hidden row and weight vector
    hidden_row = tl.load(hidden_ptr + row * H + offs, mask=offs < H, other=0.0)
    weight_vec = tl.load(weight_ptr + offs, mask=offs < H, other=0.0)

    # Compute sum of squares for the row
    x_sq = hidden_row * hidden_row
    sumsq = tl.sum(x_sq, axis=0)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Scale and store output: y = x * inv * weight
    y = hidden_row * inv * weight_vec
    tl.store(out_ptr + row * H + offs, y, mask=offs < H)


# General Triton kernel for arbitrary H: tiled two-pass approach.
@triton.jit
def tiled_scale_kernel(
    hidden_ptr,       # *float32, shape [B, H]
    weight_ptr,       # *float32, shape [H]
    out_ptr,          # *float32, shape [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # First pass: compute sum of squares across the row
    sumsq = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x_tile = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)
        x_sq = x_tile * x_tile
        sumsq += tl.sum(x_sq, axis=0)

    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute and store output y = x * inv * weight
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x_tile = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)
        w_tile = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y_tile = x_tile * inv * w_tile
        tl.store(out_ptr + row * H + offs, y_tile, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        if hidden_states.device.type != "cuda":
            hidden_states = hidden_states.to("cuda")
        if weight.device.type != "cuda":
            weight = weight.to("cuda")

        # Cast to float32 for compute
        hidden_f32 = hidden_states.to(torch.float32).contiguous()
        weight_f32 = weight.to(torch.float32).contiguous()

        B, H = hidden_f32.shape
        # Output in float32 (we'll cast to original dtype at the end)
        out = torch.empty((B, H), device=hidden_f32.device, dtype=torch.float32)

        # Launch appropriate Triton kernel
        if H == 4096:
            # One program per row
            grid = (B,)
            fullrow_scale_kernel[grid](
                hidden_f32,
                weight_f32,
                out,
                B,
                H,
                1e-5,
                4096,
                num_warps=8,
                num_stages=2,
            )
        else:
            # General tiled kernel
            BLOCK_SIZE = 256
            grid = (B,)
            tiled_scale_kernel[grid](
                hidden_f32,
                weight_f32,
                out,
                B,
                H,
                1e-5,
                BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
