import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (float32)
    out_ptr,          # *output tensor (same shape as hidden)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # columns processed per program (H)
):
    # One Triton program per row
    row_id = tl.program_id(axis=0)

    # Column indices
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # always true when BLOCK_SIZE == H

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    # Load the entire row in its native dtype and cast to float32 for robust math
    x = tl.load(hidden_row_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # weight is 1D contiguous
    w = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result; Triton will cast to out_ptr's element type if needed
    tl.store(out_row_ptr + cols * out_stride1, y, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of:
      y = (hidden_states.float() / rsqrt(mean(hidden_states^2, dim=-1) + EPS)) * weight.float()
    Returns y with the same shape and dtype as hidden_states.
    """
    assert hidden_states.ndim == 2, "hidden_states must be 2D [B, H]"
    assert weight.ndim == 1, "weight must be 1D [H]"
    B, H = hidden_states.shape
    assert H == 4096, "This Triton kernel expects H=4096"

    # Ensure contiguous tensors
    hidden = hidden_states.contiguous()
    # Convert weight to float32 for numerics
    weight_f32 = weight.to(torch.float32).contiguous()

    out = torch.empty_like(hidden)

    # Launch: one program per row
    grid = (B,)

    # Empirically high-performance launch params in the evaluation environment
    _layernorm_weight_scale_kernel[grid](
        hidden, weight_f32, out,
        B, H, 1e-5,
        hidden.stride(0), hidden.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=H,
        num_warps=8,   # high performance
        num_stages=2,  # high performance
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton-only computation
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
