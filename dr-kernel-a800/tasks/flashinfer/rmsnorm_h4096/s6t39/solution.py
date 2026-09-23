import torch
import triton
import triton.language as tl

# Triton kernel: per-row scaling using precomputed inv_rms (one scalar per row).
# One program per row. Single pass: read x, read weight, scale, write with target dtype.
@triton.jit
def _scale_kernel(
    hidden_ptr,        # *T_in, pointer to hidden_states in original dtype (cast to fp32 for compute)
    weight_ptr,        # *float32, pointer to weight (fp32)
    inv_rms_ptr,       # *float32, pointer to inv_rms (one scalar per row)
    out_ptr,           # *T_out, pointer to output (store casted result)
    hidden_size,       # int32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # Load inv_rms for this row (scalar)
    inv_rms = tl.load(inv_rms_ptr + row_id)

    # Single pass: read x and weight, compute y in fp32, cast, store
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size

        # Load x in its original dtype, cast to fp32 for compute
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0)
        x32 = x.to(tl.float32)

        # Load weight in fp32
        w32 = tl.load(weight_ptr + offs, mask=mask, other=0.0)

        # Compute
        y = x32 * inv_rms * w32

        # Cast to target dtype
        if out_dtype_code == 0:
            y_cast = y  # fp32
        elif out_dtype_code == 1:
            y_cast = y.to(tl.float16)
        elif out_dtype_code == 2:
            y_cast = y.to(tl.bfloat16)
        else:
            y_cast = y  # default fp32

        # Store (out_ptr points to tensor of desired dtype; Triton will cast as needed)
        tl.store(out_ptr + row_id * hidden_size + offs, y_cast, mask=mask)

def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096

    EPS = 1e-5

    # Ensure inputs are contiguous
    h = hidden_states.contiguous()
    w = weight.contiguous()

    # Compute in float32 for numerical stability
    h32 = h.to(torch.float32)
    w32 = w.to(torch.float32)

    # Precompute inv_rms for each row using PyTorch (1D reduction per row)
    # inv_rms[i] = 1 / sqrt(mean(h[i]^2) + eps)
    sumsq = (h32 ** 2).sum(dim=-1)  # shape [batch_size]
    mean = sumsq / hidden_size
    inv_rms = (1.0 / torch.sqrt(mean + EPS)).to(torch.float32)  # shape [batch_size]

    # Allocate output as the original input dtype to avoid an extra cast in Python
    out = torch.empty((batch_size, hidden_size), dtype=h.dtype, device=h.device)

    # Determine dtype code for casting in-kernel
    if h.dtype == torch.float32:
        out_dtype_code = 0
    elif h.dtype == torch.float16:
        out_dtype_code = 1
    elif h.dtype == torch.bfloat16:
        out_dtype_code = 2
    else:
        out_dtype_code = 0  # default fp32

    # Launch Triton kernel: one program per row
    grid = (batch_size,)
    _scale_kernel[grid](
        h32, w32, inv_rms,
        out,
        hidden_size,
        out_dtype_code=out_dtype_code,
        BLOCK_SIZE=1024,
        num_warps=8,
        num_stages=3
    )

    return out

def get_inputs():
    # For benchmarking, ensure CUDA tensors
    hidden_states = torch.randn([1, 4096], dtype=torch.bfloat16, device='cuda')
    weight = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [hidden_states, weight]

def fused_operator(tensor_0, tensor_1):
    _out = run(tensor_0, tensor_1)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
