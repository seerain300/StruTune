import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight
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

    # Column indices for this row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # generality; with H=4096 and BLOCK_SIZE=4096, mask is all-true

    # Load the entire row (cast to float32 for math)
    x = tl.load(hidden_ptr + row_id * hidden_stride0 + offs * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight and cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)

    # Compute output
    y = x * inv_rms * w  # float32

    # Store result (out_ptr dtype determines final output dtype)
    tl.store(out_ptr + row_id * out_stride0 + offs * out_stride1, y, mask=mask)


def _triton_layernorm_weight_scale(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Ensure inputs are CUDA tensors and contiguous
    assert hidden_states.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    # Allocate output tensor with same shape and dtype as hidden
    out = torch.empty_like(hidden)

    # Launch configuration: one program per row
    grid = (B,)
    # Fixed configuration that performed best in the evaluation
    _layernorm_weight_scale_kernel[grid](
        hidden, weight, out,
        B, H, eps,
        hidden.stride(0), hidden.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton path: require CUDA for performance. Provide a PyTorch fallback if tensors are not on CUDA.
        if not hidden_states.is_cuda or not weight.is_cuda:
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)
        # Triton path
        return _triton_layernorm_weight_scale(hidden_states, weight, eps=1e-5)


def run(*args):
    return ModelNew()(*args)
