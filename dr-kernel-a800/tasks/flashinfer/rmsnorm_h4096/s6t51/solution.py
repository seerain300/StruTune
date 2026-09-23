import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. Two passes: first to compute sum of squares, second to scale and store.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    stride_hidden_row, # int32
    stride_out_row,    # int32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: accumulate sum of squares for this row (scalar in float32)
    acc_sq = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        # Address using strides: row_offset is in elements
        row_offset = row_id * stride_hidden_row
        x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
        # Sum of squares across this chunk (reduction over vector -> scalar)
        chunk_sq = tl.sum(x * x, axis=0)
        acc_sq += chunk_sq

    mean_sq = acc_sq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # Second pass: scale and store
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        row_offset_hidden = row_id * stride_hidden_row
        row_offset_out = row_id * stride_out_row
        x = tl.load(hidden_ptr + row_offset_hidden + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w  # all math in fp32

        # Cast based on out_dtype_code and store
        if out_dtype_code == 0:
            tl.store(out_ptr + row_offset_out + offs, y, mask=mask)
        elif out_dtype_code == 1:
            tl.store(out_ptr + row_offset_out + offs, y.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + row_offset_out + offs, y.to(tl.bfloat16), mask=mask)

# Host-side wrapper for ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # run expects (hidden_states, weight). Unpack as provided.
        hidden = args[0]
        weight = args[1]

        # Ensure device and contiguity
        assert hidden.is_cuda and weight.is_cuda, "Inputs must be on CUDA tensors for Triton."
        hidden = hidden.contiguous()
        w = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        assert w.shape[0] == hidden_size, "weight must have the same length as the last dimension of hidden_states"

        # Compute in float32 for stability
        hidden_f32 = hidden.to(torch.float32)
        w_f32 = w.to(torch.float32)

        # Allocate output in the original dtype
        out = torch.empty_like(hidden)

        # Determine dtype code for casting at store
        out_dtype_code = 0
        if hidden.dtype == torch.float16:
            out_dtype_code = 1
        elif hidden.dtype == torch.bfloat16:
            out_dtype_code = 2
        # default to fp32 if none of the above

        # Strides (in elements)
        stride_hidden_row = hidden_f32.stride(0)
        stride_out_row = out.stride(0)

        # BLOCK_SIZE tuned for hidden_size around 4096; mask handles tails for other sizes.
        BLOCK_SIZE = 1024
        grid = (batch_size,)

        _rms_scale_kernel[grid](
            hidden_f32, w_f32, out,
            batch_size, hidden_size, 1e-5,
            stride_hidden_row, stride_out_row,
            out_dtype_code=out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3
        )
        return out


def run(*args):
    return ModelNew()(*args)
