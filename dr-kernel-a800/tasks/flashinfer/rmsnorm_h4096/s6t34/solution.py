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

    # Accumulate sum of squares
    sum_sq = 0.0
    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and write output
    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        # cast to output dtype
        if out_dtype_code == 0:
            y_cast = y
        elif out_dtype_code == 1:
            y_cast = tl.cast(y, tl.float16)
        else:
            y_cast = tl.cast(y, tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + offs, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Compute in float32 for numerical stability
        x = hidden_states.to(torch.float32).contiguous()
        w = weight.to(torch.float32).contiguous()

        # Output in fp32 and cast to original dtype at the end
        out_fp32 = torch.empty_like(hidden_states, dtype=torch.float32, device=hidden_states.device)

        # Select output dtype code
        if hidden_states.dtype == torch.float32:
            out_dtype_code = 0
        elif hidden_states.dtype == torch.float16:
            out_dtype_code = 1
        elif hidden_states.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _rms_scale_kernel[grid](
            x, w, out_fp32,
            batch_size, hidden_size, 1e-5,
            out_dtype_code,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=4
        )
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
