import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares in float32.
# Input: hidden_ptr (float32), shape [batch, hidden_size], contiguous.
# Output: out_sum_ptr[row] = sum(x[row]^2), fp32 per row.
@triton.jit
def _row_sumsq_kernel(
    hidden_ptr,     # *fp32
    out_sum_ptr,    # *fp32
    batch_size: tl.constexpr,   # we don't need it for indexing, but could use if needed
    hidden_size: tl.constexpr,  # 4096
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)

    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)

    tl.store(out_sum_ptr + row_id, acc)


# Kernel 2: compute per-row inv_rms from sum of squares.
# Input: out_sum_ptr (fp32 per row), hidden_size (constexpr), eps (float32), output inv_rms_ptr (fp32 per row).
@triton.jit
def _inv_rms_kernel(
    out_sum_ptr,    # *fp32
    inv_rms_ptr,    # *fp32
    hidden_size: tl.constexpr,  # 4096
    eps,               # float32
    BLOCK_SIZE: tl.constexpr  # dummy, not used
):
    row_id = tl.program_id(0)
    sumsq = tl.load(out_sum_ptr + row_id)
    mean = sumsq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    tl.store(inv_rms_ptr + row_id, inv_rms)


# Kernel 3: scale each row using inv_rms and weight, write output in original dtype.
@triton.jit
def _scale_write_kernel(
    hidden_ptr,      # *fp32
    weight_ptr,      # *fp32
    out_ptr,         # *T_out (we'll cast in kernel)
    inv_rms_ptr,     # *fp32
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    inv_rms = tl.load(inv_rms_ptr + row_id)

    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        if out_dtype_code == 0:
            y_cast = y
        elif out_dtype_code == 1:
            y_cast = tl.cast(y, tl.float16)
        else:
            y_cast = tl.cast(y, tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + offs, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Expect hidden_states: [batch_size, 4096], weight: [hidden_size]
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch_size, hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Ensure contiguous and cast to fp32 for compute
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)

        # Output tensor in the original dtype
        out = torch.empty_like(hidden_states)

        # Determine output dtype code
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: store as fp32 (rare case)
            out = torch.empty_like(hidden_states, dtype=torch.float32)
            out_dtype_code = 0

        # Allocate per-row sums and inv_rms
        out_sum = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)
        inv_rms = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)

        # Launch kernel 1: compute per-row sum of squares
        BLOCK_SIZE = 1024  # tile size; 4 iterations for 4096
        grid = (batch_size,)
        _row_sumsq_kernel[grid](
            hidden, out_sum,
            hidden_size=hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3
        )

        # Launch kernel 2: compute inv_rms per row (Triton kernel, no host torch ops)
        _inv_rms_kernel[grid](
            out_sum, inv_rms,
            hidden_size=hidden_size,
            eps=1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,   # simple scalar per row, minimal resources
            num_stages=1
        )

        # Launch kernel 3: scale and write output
        _scale_write_kernel[grid](
            hidden, weight, out, inv_rms,
            batch_size=batch_size,
            hidden_size=hidden_size,
            out_dtype_code=out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3
        )

        return out


def run(*args):
    return ModelNew()(*args)
