import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel specialized for H == 4096: one program per row.
@triton.jit
def fullrow_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                          B, H, EPS,
                          BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Compute linear offsets for the row
    offsets = row_id * H + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * H  # always true for BLOCK_SIZE == H, but keep mask for safety

    # Load hidden row and weight vector
    x = tl.load(hidden_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute sum of squares
    x_sq = x * x
    sumsq = tl.sum(x_sq, axis=0)

    # Compute inv_rms
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Compute output: y = x * inv * w
    y = x * inv * w

    # Store output
    tl.store(out_ptr + offsets, y, mask=mask)


# General Triton kernel for arbitrary H: two-pass, tiled across columns.
@triton.jit
def tiled_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                        B, H, EPS,
                        BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # First pass: compute sum of squares for this row
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row_id * H + cols, mask=mask, other=0.0)
        x_sq = x * x
        # Reduce tile to scalar and accumulate
        sumsq += tl.sum(x_sq, axis=0)

    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute and store output
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row_id * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA and cast to float32 for computation
        if hidden_states.device.type != "cuda" or not TRITON_AVAILABLE:
            # Fallback to PyTorch computation if Triton/CUDA not available
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Prepare inputs: make contiguous and cast to float32
        hidden_f32 = hidden_states.to(torch.float32).contiguous()
        weight_f32 = weight.to(torch.float32).contiguous()
        B, H = hidden_f32.shape
        out = torch.empty((B, H), device=hidden_f32.device, dtype=torch.float32)

        if H == 4096:
            # Launch fullrow kernel: one program per row, process entire row
            grid = (B,)
            fullrow_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H, 1e-5,
                BLOCK_SIZE=4096,
                num_warps=8,  # tuned for 4096-wide row
                num_stages=2,
            )
        else:
            # Launch tiled kernel: general fallback
            grid = (B,)
            tiled_scale_kernel[grid](
                hidden_f32, weight_f32, out,
                B, H, 1e-5,
                BLOCK_SIZE=256,
                num_warps=4,
                num_stages=2,
            )

        # Cast to original dtype for return
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
