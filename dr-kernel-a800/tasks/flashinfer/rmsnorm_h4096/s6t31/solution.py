import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8, num_stages=3),
    ],
    key=['hidden_size'],  # tune based on the row length
)
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # tile size (autotuned)
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: accumulate sum of squares over the row.
    sum_sq = 0.0
    num_tiles = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile in range(0, num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)

    # Compute inv_rms = 1 / sqrt(mean(x^2) + eps) where mean = sum_sq / hidden_size
    mean = sum_sq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and write output
    for tile in range(0, num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        # Cast to output dtype before store
        if out_dtype_code == 0:
            y_cast = y  # fp32
        elif out_dtype_code == 1:
            y_cast = y.to(tl.float16)
        else:  # out_dtype_code == 2 (bf16)
            y_cast = y.to(tl.bfloat16)
        tl.store(out_ptr + row_id * hidden_size + offs, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure device and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA for Triton kernel"
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
        assert weight.dim() == 1, "weight must be 1D [hidden_size]"
        hidden_size = hidden_states.shape[1]
        assert hidden_size == 4096, "This optimized Triton kernel expects hidden_size == 4096"

        # Cast inputs to float32 for computation
        hidden_in = hidden_states.contiguous().to(torch.float32)
        weight_in = weight.contiguous().to(torch.float32)

        batch_size = hidden_states.shape[0]
        # Output in the same dtype as input hidden_states
        out = torch.empty((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)

        # Prepare dtype code: 0=fp32, 1=fp16, 2=bf16
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Fallback: store as fp32 if some other dtype appears (unlikely here)
            out_dtype_code = 0

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _rms_scale_kernel[grid](
            hidden_in, weight_in, out,
            batch_size, hidden_size, 1e-5,
            out_dtype_code
        )
        return out


def run(*args):
    return ModelNew()(*args)
