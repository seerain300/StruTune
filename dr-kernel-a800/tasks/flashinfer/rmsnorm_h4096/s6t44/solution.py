import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (store casted result)
    hidden_size,       # int32, e.g., 4096
    eps,               # float32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: compute sum of squares to derive inv_rms
    sumsq = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        # Provide alignment/multiplicity hints to help vectorization
        tl.max_contiguous(offs, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + eps)

    # Second pass: compute output and store.
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        tl.max_contiguous(offs, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w

        # Cast to desired output dtype
        if out_dtype_code == 0:
            out_val = y
        elif out_dtype_code == 1:  # fp16
            out_val = y.to(tl.float16)
        elif out_dtype_code == 2:  # bf16
            out_val = y.to(tl.bfloat16)
        else:
            out_val = y  # default to fp32

        tl.store(out_ptr + row_id * hidden_size + offs, out_val, mask=mask)

# Helper to determine output dtype code
def _out_dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure shape; no tensor math on host
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
        assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1], "weight must be 1D of length hidden_size"
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        assert hidden_size == 4096, "This optimized kernel assumes hidden_size == 4096"

        # Cast inputs to float32 for compute (avoid .contiguous() to prevent unnecessary copies)
        hidden = hidden_states.to(torch.float32)  # [batch, 4096], float32
        weight = weight.to(torch.float32)        # [4096], float32

        # Allocate output with same shape as hidden_states (original dtype)
        out = torch.empty_like(hidden_states)

        eps = 1e-5
        out_dtype_code = _out_dtype_code(out.dtype)  # 2 for bfloat16
        BLOCK_SIZE = 1024

        # Launch Triton kernel: one program per row
        grid = (batch_size,)

        _rms_scale_kernel[grid](
            hidden, weight, out,
            hidden_size, eps,
            out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,   # good default for 1024 block size
            num_stages=3   # pipeline stages for better latency hiding
        )

        return out


def run(*args):
    return ModelNew()(*args)
