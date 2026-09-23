import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *float32, pointer to output (computed in fp32)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: accumulate sum of squares across the row
    acc = 0.0  # scalar accumulator in fp32
    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + idx, mask=mask, other=0.0)
        # x is float32; accumulate sum of squares
        acc += tl.sum(x * x, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + eps)
    mean = acc / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and store output
    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + idx, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0)  # weight is length hidden_size
        y = x * inv_rms * w
        # Store to out_ptr (fp32). If the caller wants a different dtype, cast after kernel.
        tl.store(out_ptr + row_id * hidden_size + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure on CUDA device and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous().to(torch.float32)   # compute in fp32
        weight = weight.contiguous().to(torch.float32)

        batch_size, hidden_size = hidden.shape

        # Output computed in fp32 for numerical stability
        out_fp32 = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        BLOCK_SIZE = 1024  # 4 iterations for hidden_size=4096

        _rms_scale_kernel[grid](
            hidden, weight, out_fp32,
            batch_size, hidden_size, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3
        )

        # Cast output back to original hidden_states dtype
        out = out_fp32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
