import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel specialized for H == 4096: process an entire row in one go.
# One program per row. Computes sum of squares, inv_rms, then outputs y = x * inv_rms * weight.
@triton.jit
def fullrow_scale_kernel(
    hidden_ptr,      # *f32, pointer to hidden_states (float32) of shape [B, H]
    weight_ptr,      # *f32, pointer to weight (float32) of shape [H]
    out_ptr,         # *f32, pointer to output (float32) of shape [B, H]
    B,               # int32: batch size (not used but kept for signature symmetry)
    H: tl.constexpr, # hidden size (4096)
    EPS,             # float32 epsilon
    BLOCK_SIZE: tl.constexpr,  # 4096
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    hidden_row_ptr = hidden_ptr + row * H + cols
    out_row_ptr = out_ptr + row * H + cols
    # Load x row
    x = tl.load(hidden_row_ptr, mask=cols < H, other=0.0)
    # Sum of squares
    sumsq = tl.sum(x * x, axis=0)
    # inv_rms = 1 / sqrt((sumsq / H) + EPS)
    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    # Load weight and compute output
    w = tl.load(weight_ptr + cols, mask=cols < H, other=0.0)
    y = x * inv_rms * w
    tl.store(out_row_ptr, y, mask=cols < H)


# Triton kernel general fallback for arbitrary H (tiled, two-pass).
@triton.jit
def tiled_scale_kernel(
    hidden_ptr,      # *f32, pointer to hidden_states (float32), shape [B, H]
    weight_ptr,      # *f32, pointer to weight (float32), shape [H]
    out_ptr,         # *f32, pointer to output (float32), shape [B, H]
    B,               # int32 batch size
    H,               # int32 hidden size
    EPS,             # float32 epsilon
    BLOCK_SIZE: tl.constexpr,  # tile size over columns (e.g., 256)
):
    row = tl.program_id(0)
    # First pass: accumulate sum of squares across the row
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    # Second pass: compute output and store
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA; cast to float32 for stable math.
        # The provided get_inputs returns CUDA tensors, but we enforce device handling here.
        if hidden_states.device.type != "cuda" or weight.device.type != "cuda":
            # Fallback: compute with torch if not CUDA (evaluation typically provides CUDA tensors).
            hidden_f32 = hidden_states.float()
            weight_f32 = weight.float()
            B, H = hidden_f32.shape
            inv_rms = 1.0 / torch.sqrt((hidden_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-5))
            out = (hidden_f32 * inv_rms) * weight_f32
            return out.to(hidden_states.dtype)
        hidden_f32 = hidden_states.float().contiguous()
        weight_f32 = weight.float().contiguous()
        B, H = hidden_f32.shape
        # Output tensor (float32)
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_f32.device)

        # Choose kernel
        if H == 4096:
            grid = (B,)
            fullrow_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H, 1e-5,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=2,
            )
        else:
            grid = (B,)
            BLOCK_SIZE = 256
            tiled_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H, 1e-5,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )

        # Return cast to original dtype
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
