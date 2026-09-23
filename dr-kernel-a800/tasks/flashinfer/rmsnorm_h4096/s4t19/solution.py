import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (e.g., bfloat16), cast to float32 in-kernel
    out_ptr,          # *output (will match the dtype of hidden_ptr)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # 4096 for this task
):
    # One program per row
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    # H should equal BLOCK_SIZE for this task; mask kept for safety
    mask = cols < H

    # Base offsets for this row
    hidden_row_off = row_id * hidden_stride0
    out_row_off = row_id * out_stride0

    # Load hidden row and cast to float32 for math
    x_ptrs = hidden_ptr + hidden_row_off + cols * hidden_stride1
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_f32 = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    x2 = x_f32 * x_f32
    sumsq = tl.sum(x2, axis=0)  # scalar
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w_f32 = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y_f32 = x_f32 * inv_rms * w_f32

    # Store output (Triton will cast to the pointer dtype as needed)
    out_ptrs = out_ptr + out_row_off + cols * out_stride1
    tl.store(out_ptrs, y_f32, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Triton requires CUDA tensors"
        # Ensure contiguous layout for simple 2D indexing
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output tensor (same dtype as hidden for parity with original)
        out = torch.empty_like(hidden)

        # Kernel launch configuration: one program per row, process 4096 columns
        grid = (B,)
        BLOCK_SIZE = 4096
        # Best-performing params in prior evaluations
        num_warps = 8
        num_stages = 2

        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, self.eps,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
