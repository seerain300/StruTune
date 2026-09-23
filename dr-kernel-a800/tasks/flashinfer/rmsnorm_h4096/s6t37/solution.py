import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: accumulate sum of squares across the row
    sumsq = 0.0  # scalar float32
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + col_offsets, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + eps)  # 1.0 / sqrt(mean + eps)

    # Second pass: read x again, scale, and write output
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + col_offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
        y = x * inv_rms * w  # compute in float32

        # Cast to target dtype at store
        if out_dtype_code == 0:
            y_cast = y
        elif out_dtype_code == 1:
            y_cast = tl.cast(y, tl.float16)
        else:
            y_cast = tl.cast(y, tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + col_offsets, y_cast, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous float32 compute
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Cast inputs to float32 for compute (contiguous)
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)

        # Prepare output with target dtype (match hidden_states dtype)
        out = torch.empty_like(hidden_states, device=hidden_states.device)

        # Encode output dtype for kernel
        dtype_code = 0  # fp32 default
        if out.dtype == torch.float16:
            dtype_code = 1
        elif out.dtype == torch.bfloat16:
            dtype_code = 2
        else:
            # default to fp32 if some other dtype appears
            dtype_code = 0

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _rms_scale_kernel[grid](
            hidden, weight, out,
            batch_size, hidden_size, 1e-5,
            dtype_code,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=3,
        )
        return out


def run(*args):
    return ModelNew()(*args)
