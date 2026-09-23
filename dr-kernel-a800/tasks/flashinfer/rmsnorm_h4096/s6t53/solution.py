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
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row
    # First pass: compute sum of squares across the row
    acc = 0.0
    # Loop over chunks of size BLOCK_SIZE
    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    mean = acc / hidden_size
    inv_rms = tl.rsqrt(mean + eps)

    # Second pass: scale and store
    for start in range(0, hidden_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row_id * hidden_size + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "ModelNew requires CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        # Compute in float32, output in original dtype
        hidden_f32 = hidden.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        out = torch.empty_like(hidden, dtype=hidden.dtype)

        eps = 1e-5
        BLOCK_SIZE = 1024  # good default for 4096; can experiment with 512/2048
        grid = (batch_size,)

        _rms_scale_kernel[grid](
            hidden_f32,
            weight_f32,
            out,
            batch_size,
            hidden_size,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3,
        )
        return out


def run(*args):
    return ModelNew()(*args)
