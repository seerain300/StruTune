import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Full-row fused kernel: assumes H <= BLOCK_SIZE and BLOCK_SIZE is a multiple of 128 for good perf.
# One program per row. Loads entire row x[row, :], computes sumsq, inv_rms, then y = x * inv * w, and stores.
@triton.jit
def fullrow_scale_kernel(x_ptr, w_ptr, out_ptr,
                          B, H,
                          EPS: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Row base pointers
    x_row = x_ptr + row_id * H
    out_row = out_ptr + row_id * H

    # Vector of column offsets
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load row and weight vector
    x = tl.load(x_row + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)

    # Compute sum of squares for this row
    x2 = x * x
    sumsq = tl.sum(x2, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Scale and store
    y = x * inv * w
    tl.store(out_row + offs, y, mask=mask)


# General tiled kernel: works for any H. Two-pass approach (sum then output).
@triton.jit
def tiled_scale_kernel(x_ptr, w_ptr, out_ptr,
                        B, H,
                        EPS: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    x_row = x_ptr + row_id * H
    out_row = out_ptr + row_id * H

    # Pass 1: compute sum of squares per row
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_row + offs, mask=mask, other=0.0)
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)

    # Compute inv_rms
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Pass 2: compute y = x * inv * w and store
    for col_start in range(0, H, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_row + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(out_row + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA and float32 for computation
        device = hidden_states.device
        B, H = hidden_states.shape
        assert weight.shape == (H,), f"weight must have shape [{H}], got {tuple(weight.shape)}"
        # Cast inputs to float32 for numerical stability; keep original dtype for final output
        hidden_f32 = hidden_states.to(torch.float32).contiguous()
        weight_f32 = weight.to(torch.float32).contiguous()

        # Allocate output tensor in float32
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Choose kernel: fullrow for H <= 4096 and divisible by 4096, else tiled
        use_fullrow = (H <= 4096) and (H % 4096 == 0)

        if use_fullrow:
            grid = (B,)
            fullrow_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H,
                EPS=1e-5,
                BLOCK_SIZE=4096,
                num_warps=8,  # more warps for 4096-wide vector
                num_stages=2,
            )
        else:
            grid = (B,)
            tiled_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H,
                EPS=1e-5,
                BLOCK_SIZE=256,
                num_warps=4,
                num_stages=2,
            )

        # Cast back to original hidden_states dtype before returning
        out = out.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
