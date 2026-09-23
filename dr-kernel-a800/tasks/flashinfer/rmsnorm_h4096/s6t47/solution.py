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

    # First pass: accumulate sum of squares to compute inv_rms
    sum_sq = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean_sq = sum_sq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # Second pass: scale and write output
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms
        y = y * w
        tl.store(out_ptr + row_id * hidden_size + offs, y, mask=mask)

def run(hidden_states, weight):
    # hidden_states: [batch_size, 4096], weight: [4096], original dtypes (e.g., bfloat16)
    # Compute in float32, store back in original dtype.
    assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096, "hidden_size must be 4096"

    # Cast inputs to float32 for compute; do not copy if already contiguous
    hidden32 = hidden_states.to(torch.float32)
    weight32 = weight.to(torch.float32)

    out32 = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=hidden_states.device)

    # Launch Triton kernel: one program per row
    BLOCK_SIZE = 1024  # 4 iterations over 4096
    num_warps = 8
    num_stages = 3
    grid = (batch_size,)

    _rms_scale_kernel[grid](
        hidden32, weight32, out32,
        batch_size, hidden_size, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, num_stages=num_stages
    )

    # Cast back to original dtype (e.g., bfloat16) to match the original model's output dtype
    return out32.to(hidden_states.dtype)

# The following are unchanged from the original for testing convenience.
def get_inputs():
    hidden_states = torch.randn([1, 4096], dtype=torch.bfloat16, device='cuda')
    weight = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [hidden_states, weight]

def fused_operator(tensor_0, tensor_1):
    _out = run(tensor_0, tensor_1)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Triton-only computation: no torch ops in forward
        assert len(args) == 2, "forward expects hidden_states and weight"
        return run(*args)


def run(*args):
    return ModelNew()(*args)
