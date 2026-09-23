import torch
import triton
import triton.language as tl

# Triton kernel: per-row sum of squares in float32. One program per row.
@triton.jit
def _row_sumsq_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    sum_ptr,           # *float32, output: per-row sum of squares
    batch_size,        # int32
    hidden_size,       # int32
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    # Accumulate sum of squares across the row in fp32
    sum_val = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        x2 = x * x
        sum_val += tl.sum(x2, axis=0)
    tl.store(sum_ptr + row_id, sum_val)


# Triton kernel: per-row scaling using inv_rms[row] and weight, store to output dtype.
@triton.jit
def _scale_store_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    invrms_ptr,        # *float32, pointer to per-row inv_rms (shape [batch_size])
    out_ptr,           # *T_out, pointer to output tensor
    batch_size,        # int32
    hidden_size,       # int32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row
    inv_rms = tl.load(invrms_ptr + row_id)  # scalar per row in fp32
    for col in range(0, hidden_size, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        # Cast to output dtype based on code
        if out_dtype_code == 0:
            y_cast = y
        elif out_dtype_code == 1:
            y_cast = y.to(tl.float16)
        else:
            y_cast = y.to(tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + cols, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Cast compute to float32
        hidden_fp32 = hidden.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # Allocate per-row sum of squares (fp32)
        sum_sq = torch.empty((batch_size,), device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel to compute sum of squares per row
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        _row_sumsq_kernel[grid](
            hidden_fp32, sum_sq, batch_size, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3
        )

        # Compute mean and inv_rms per row on host (CPU scalar math, allowed):
        # inv_rms = 1 / sqrt(mean(x^2) + eps)
        eps = 1e-5
        mean = sum_sq / hidden_size
        inv_rms = torch.rsqrt(mean + eps)  # shape [batch_size], fp32

        # Prepare output tensor in original dtype
        out = torch.empty_like(hidden_states)

        # Determine out dtype code
        out_dtype = out.dtype
        if out_dtype == torch.float32:
            out_dtype_code = 0
        elif out_dtype == torch.float16:
            out_dtype_code = 1
        elif out_dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: cast to fp32 output
            out = torch.empty_like(hidden_states, dtype=torch.float32)
            out_dtype_code = 0

        # Launch Triton scaling kernel: y = x * inv_rms[row] * weight
        _scale_store_kernel[grid](
            hidden_fp32, weight_fp32, inv_rms, out, batch_size, hidden_size,
            out_dtype_code=out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3
        )

        return out


# Provided get_inputs helper (unchanged)
def get_inputs():
    hidden_states = torch.randn([1, 4096], dtype=torch.bfloat16, device='cuda')
    weight = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [hidden_states, weight]


def run(*args):
    return ModelNew()(*args)
