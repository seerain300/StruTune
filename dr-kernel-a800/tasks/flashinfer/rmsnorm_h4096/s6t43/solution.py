import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares in float32. Output: sum_sq[row] = sum(x[row]^2)
@triton.jit
def _sum_of_squares_row(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    sum_ptr,           # *float32, output per-row sum
    batch_size,        # int32
    hidden_size,       # int32
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    # Guard: if row_id >= batch_size, return
    if row_id >= batch_size:
        return
    # Accumulate sum of squares over the row
    sum_val = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        # x is float32 (input was cast to float32 before kernel launch)
        sum_val += tl.sum(x * x, axis=0)
    tl.store(sum_ptr + row_id, sum_val)


# Kernel 2: compute inv_rms per row using rsqrt: inv_rms[row] = rsqrt(sum_sq[row] / hidden_size + eps)
@triton.jit
def _compute_inv_rms_row(
    sum_ptr,           # *float32, per-row sum of squares
    inv_ptr,           # *float32, per-row inv_rms
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
):
    row_id = tl.program_id(0)
    if row_id >= batch_size:
        return
    sum_val = tl.load(sum_ptr + row_id)
    mean = sum_val / hidden_size
    inv_rms = tl.rsqrt(mean + eps)
    tl.store(inv_ptr + row_id, inv_rms)


# Kernel 3: scale and store output: y = x * inv_rms[row] * weight
@triton.jit
def _scale_and_store_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    inv_ptr,           # *float32, per-row inv_rms
    batch_size,        # int32
    hidden_size,       # int32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)
    if row_id >= batch_size:
        return
    inv_rms = tl.load(inv_ptr + row_id)  # scalar float32
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)  # float32
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)                         # float32
        y = x * inv_rms * w
        # Cast to output dtype
        if out_dtype_code == 0:
            y_cast = y
        elif out_dtype_code == 1:
            y_cast = y.to(tl.float16)
        else:
            y_cast = y.to(tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + offs, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        # Constants from the original code
        EPS = 1e-5
        # Compute in float32 for stability
        hidden_f32 = hidden.to(torch.float32)

        # Allocate intermediates on device
        sum_sq = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)
        inv_rms = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)

        # Kernel 1: sum of squares per row
        grid_sum = (batch_size,)
        _sum_of_squares_row[grid_sum](
            hidden_f32, sum_sq,
            batch_size, hidden_size,
            BLOCK_SIZE=1024,
            num_warps=8, num_stages=3
        )

        # Kernel 2: compute inv_rms per row using rsqrt
        _compute_inv_rms_row[grid_sum](
            sum_sq, inv_rms,
            batch_size, hidden_size,
            EPS,
            num_warps=4, num_stages=2
        )

        # Prepare output tensor with original dtype
        out = torch.empty_like(hidden)  # dtype matches hidden_states
        # Map dtype to code for kernel
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: store as float32
            out_dtype_code = 0

        # Kernel 3: scale and store
        _scale_and_store_kernel[(batch_size,)](
            hidden_f32, weight.to(torch.float32), out,
            inv_rms,
            batch_size, hidden_size,
            out_dtype_code=out_dtype_code,
            BLOCK_SIZE=1024,
            num_warps=8, num_stages=3
        )

        return out


def run(*args):
    return ModelNew()(*args)
