import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16)
    weight_ptr,       # *const weight tensor (e.g., bfloat16), we cast to float32 in-kernel
    out_ptr,          # *output tensor
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon (1e-5)
    hidden_stride0: tl.int32,  # stride along batch dim
    hidden_stride1: tl.int32,  # stride along hidden dim
    out_stride0: tl.int32,     # stride along batch dim
    out_stride1: tl.int32,     # stride along hidden dim
    BLOCK_SIZE: tl.constexpr,  # columns processed per program (here H=4096)
):
    # Each program handles one row
    row_id = tl.program_id(0)
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # BLOCK_SIZE==H in our case, but keep for generality

    # Load hidden row and cast to float32 for math
    hidden_row = tl.load(hidden_row_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(hidden_row, tl.float32)

    # Compute sum of squares and inv_rms
    sumsq = tl.sum(x * x, axis=0)  # scalar
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar

    # Load weight vector as float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)  # ensure float32 math

    # Compute output
    y = x * inv_rms * w

    # Store result
    tl.store(out_row_ptr + cols * out_stride1, y, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous for simple stride math
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    out = torch.empty_like(hidden)

    # Launch grid: one program per row
    grid = (B,)

    # Empirically best-performing launch params in the evaluation environment
    _layernorm_weight_scale_kernel[grid](
        hidden, weight, out,
        B, H, 1e-5,
        hidden.stride(0), hidden.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=H,
        num_warps=8,   # best-performing
        num_stages=2,  # best-performing
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton-only computation
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
