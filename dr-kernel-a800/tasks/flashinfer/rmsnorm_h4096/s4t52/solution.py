import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (cast to float32 in-kernel)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    # One program per row
    row_id = tl.program_id(0)

    # Column indices for this row (BLOCK_SIZE must be >= H; here H=4096)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # keeps generality; for H=4096, this is all True

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    # Load hidden row and cast to float32 for math
    x = tl.load(hidden_row_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x)

    # Compute inv_rms = rsqrt(mean + EPS) with mean = sumsq / H
    H_f = tl.cast(H, tl.float32)
    mean = sumsq / H_f
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result
    tl.store(out_row_ptr + cols * out_stride1, y, mask=mask)


def _triton_run(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Ensure CUDA and contiguity
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    # Output tensor with same shape and dtype as hidden
    out = torch.empty_like(hidden)

    # Launch parameters: one program per row
    grid = (B,)

    # Strides (elements, not bytes)
    hidden_stride0 = hidden.stride(0)
    hidden_stride1 = hidden.stride(1)
    out_stride0 = out.stride(0)
    out_stride1 = out.stride(1)

    # Launch Triton kernel
    _layernorm_weight_scale_kernel[grid](
        hidden, weight, out,
        B, H, eps,
        hidden_stride0, hidden_stride1,
        out_stride0, out_stride1,
        BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton-only computation: no torch ops on tensors inside forward
        return _triton_run(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
