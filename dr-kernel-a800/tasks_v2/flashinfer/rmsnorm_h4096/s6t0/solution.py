import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight
# Assumes hidden_size == 4096 for this implementation.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16 (for casting in kernel)
    BLOCK_SIZE: tl.constexpr        # e.g., 128
):
    row_id = tl.program_id(0)  # one program per row
    # Safety: if row_id >= batch_size, exit (normally grid matches batch_size)
    # (We still guard loads/stores with masks, but this is defensive.)
    if row_id >= batch_size:
        return

    # Compute base pointer for this row
    # hidden_ptr has shape [batch_size, hidden_size], row_id * hidden_size offset
    row_base = hidden_ptr + row_id * hidden_size

    # First pass: accumulate sum of squares across columns in chunks
    sumsq = 0.0
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(row_base + cols, mask=mask, other=0.0)
        # x is float32
        sumsq += tl.sum(x * x, axis=0)

    # mean = sumsq / hidden_size
    mean = sumsq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)  # float32 scalar

    # Second pass: compute output and store
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        x = tl.load(row_base + cols, mask=mask, other=0.0)  # float32
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # float32
        y = x * inv_rms * w  # float32

        # Cast to output dtype based on out_dtype_code
        # 0: fp32, 1: fp16, 2: bf16
        # Triton allows casting to tl.float16 or tl.bfloat16 via tl.cast.
        y_cast = y  # keep as fp32 for now; will cast before store
        if out_dtype_code == 1:
            y_cast = tl.cast(y, tl.float16)
        elif out_dtype_code == 2:
            y_cast = tl.cast(y, tl.bfloat16)
        # else 0: fp32, keep as is

        out_row_base = out_ptr + row_id * hidden_size
        tl.store(out_row_base + cols, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        - Computes RMS per row in float32, scales by weight, returns in original dtype.
        - Assumes hidden_size == 4096 (as in the original code).
        """
        # Ensure on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "ModelNew requires CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Original code asserts hidden_size == 4096
        hidden_size = hidden.shape[1]
        assert hidden_size == 4096, f"Expected hidden_size == 4096, got {hidden_size}"

        # Compute in float32 (as original)
        x = hidden.to(torch.float32)
        w = weight.to(torch.float32)

        batch_size = hidden.shape[0]
        EPS = 1e-5

        # Output tensor in original dtype
        out = torch.empty_like(hidden)  # same shape, original dtype (e.g., bfloat16)

        # Determine output dtype code for kernel casting
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: convert output to float32 if unexpected dtype
            out = torch.empty_like(hidden, dtype=torch.float32)
            out_dtype_code = 0

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Choose BLOCK_SIZE; 128 works well for 4096 columns
        BLOCK_SIZE = 128
        num_warps = 4  # reasonable default; 4 or 8 typically fine for such sizes
        num_stages = 2

        _rms_scale_kernel[grid](
            x, w, out,
            batch_size, hidden_size,
            EPS,
            out_dtype_code=out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return out


def run(*args):
    return ModelNew()(*args)
