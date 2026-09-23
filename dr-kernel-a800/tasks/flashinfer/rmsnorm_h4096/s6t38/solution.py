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
    sumsq = 0.0
    for start in range(0, hidden_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offsets, mask=mask, other=0.0)
        sumsq += tl.sum(x * x)

    mean = sumsq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and store
    for start in range(0, hidden_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        y = x * inv_rms * w
        # Cast based on out_dtype_code
        if out_dtype_code == 0:
            y_cast = y  # fp32
        elif out_dtype_code == 1:
            y_cast = tl.cast(y, tl.float16)  # fp16
        else:
            y_cast = tl.cast(y, tl.bfloat16)  # bf16
        tl.store(out_ptr + row_id * hidden_size + offsets, y_cast, mask=mask)


def _to_out_dtype_code(t: torch.Tensor) -> int:
    if t.dtype == torch.float32:
        return 0
    elif t.dtype == torch.float16:
        return 1
    elif t.dtype == torch.bfloat16:
        return 2
    else:
        # default to fp32 if some other dtype appears
        return 0


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous float32 compute
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)
        batch_size = hidden.shape[0]
        hidden_size = hidden.shape[1]
        # Output tensor in original dtype
        out = torch.empty_like(hidden_states)

        # Choose dtype code for store
        dtype_code = _to_out_dtype_code(hidden_states)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Execution parameters tuned for hidden_size=4096
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
