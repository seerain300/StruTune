import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Full-row Triton kernel: H == 4096
# One program per row. Loads the whole row of hidden_states, computes sumsq,
# inv_rms, and immediately produces y = x * inv_rms * weight, storing to out.
@triton.jit
def fullrow_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                          B, H, EPS,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    # Base offsets for this row
    row_off = row * H

    # Load the entire row into a vector
    cols = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE == H == 4096
    # Compute pointers for this row
    x_ptrs = hidden_ptr + row_off + cols
    w_ptrs = weight_ptr + cols

    # Mask: valid since BLOCK_SIZE == H
    x = tl.load(x_ptrs)
    w = tl.load(w_ptrs)

    # Compute sum of squares for this row
    sq = x * x
    sumsq = tl.sum(sq, axis=0)

    # Compute inv_rms: 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms
    y = y * w

    # Store output row
    out_row_ptrs = out_ptr + row_off + cols
    tl.store(out_row_ptrs, y)


# General tiled Triton kernel: arbitrary H
# One program per row. Two-pass: first pass computes sumsq; second pass computes output.
@triton.jit
def tiled_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                        B, H, EPS,
                        BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_off = row * H
    sumsq = 0.0  # scalar accumulator for this row

    # First pass: compute sum of squares across tiles
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row_off + cols, mask=mask, other=0.0)
        sq = x * x
        # Reduce this tile to scalar and accumulate
        sumsq += tl.sum(sq, axis=0)

    # Compute inv_rms
    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute output and store
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row_off + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms
        y = y * w
        tl.store(out_ptr + row_off + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguous
        assert hidden_states.dim() == 2, "hidden_states must be [B, H]"
        assert weight.dim() == 1, "weight must be [H]"
        H = hidden_states.shape[1]
        B = hidden_states.shape[0]
        assert H == weight.shape[0], "weight length must equal hidden_states' last dim"
        assert hidden_states.is_cuda and weight.is_cuda, "inputs must be on CUDA device"

        # Cast to float32 and make contiguous for Triton
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        weight_f32 = weight.contiguous().to(torch.float32)

        # Output tensor (float32 for computation)
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_f32.device)

        # Launch appropriate Triton kernel
        if H == 4096:
            # Fullrow specialized kernel: single pass, no second read of hidden_states
            grid = (B,)
            fullrow_scale_kernel[grid](hidden_f32, weight_f32, out, B, H, 1e-5, BLOCK_SIZE=4096, num_warps=8, num_stages=2)
        else:
            # General tiled kernel
            grid = (B,)
            tiled_scale_kernel[grid](hidden_f32, weight_f32, out, B, H, 1e-5, BLOCK_SIZE=256, num_warps=4, num_stages=2)

        # Cast to original dtype for return (host-side cast, not in Triton)
        out = out.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
