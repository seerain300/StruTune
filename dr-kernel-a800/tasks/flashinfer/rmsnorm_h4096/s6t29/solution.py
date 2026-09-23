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

    # Accumulator for sum of squares (float32)
    sum_sq = 0.0

    # First pass: compute sum of squares across the row
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean_sq = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Second pass: scale and write output
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w

        # Store with dtype control
        if out_dtype_code == 0:
            # fp32
            tl.store(out_ptr + row_id * hidden_size + cols, y, mask=mask)
        elif out_dtype_code == 1:
            # fp16
            tl.store(out_ptr + row_id * hidden_size + cols, y.to(tl.float16), mask=mask)
        else:
            # bf16
            tl.store(out_ptr + row_id * hidden_size + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If not CUDA, fall back to original PyTorch behavior for correctness
        if (not hidden_states.is_cuda) or (not weight.is_cuda):
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguity and compute in float32
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)

        batch_size = hidden.shape[0]
        hidden_size = hidden.shape[1]
        assert hidden_size == 4096, "This optimized kernel expects hidden_size == 4096."

        # Output tensor with same dtype as input hidden_states
        out = torch.empty_like(hidden, dtype=hidden_states.dtype)

        # Encode output dtype for the kernel
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: compute in float32 and cast after
            out = torch.empty_like(hidden, dtype=torch.float32)
            out_dtype_code = 0

        grid = (batch_size,)
        # Launch Triton kernel
        _rms_scale_kernel[grid](
            hidden, weight, out,
            batch_size, hidden_size, 1e-5,
            out_dtype_code,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=3,
        )

        return out


def run(*args):
    return ModelNew()(*args)
