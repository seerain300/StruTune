import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16)
    weight_ptr,       # *const weight tensor (same length as hidden's last dim)
    out_ptr,          # *output tensor (same shape as hidden; dtype from allocation)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (fixed 4096 in this task)
    EPS: tl.float32,  # epsilon
):
    # One program per row
    row_id = tl.program_id(0)
    # Base pointers for this row
    row_hidden_ptr = hidden_ptr + row_id * H
    row_out_ptr = out_ptr + row_id * H

    # Column indices
    cols = tl.arange(0, H)

    # Load row and weight; Triton will load in the pointer's element type
    x = tl.load(row_hidden_ptr + cols)
    w = tl.load(weight_ptr + cols)

    # Cast to float32 for numerically stable computation
    x32 = tl.cast(x, tl.float32)
    w32 = tl.cast(w, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute mean and inv_rms
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Compute output: y = x * inv_rms * w
    y32 = x32 * inv_rms * w32

    # Store to output; Triton will cast y32 to the dtype of out_ptr if needed
    tl.store(row_out_ptr + cols, y32)


def _triton_layernorm_weight_scale(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Ensure inputs are CUDA tensors and contiguous
    assert hidden_states.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    # Output tensor with same shape and dtype as hidden
    out = torch.empty_like(hidden)

    # Launch one program per row
    grid = (B,)

    _layernorm_weight_scale_kernel[grid](
        hidden, weight, out,
        B, H, eps,
        num_warps=8,  # tuned for performance on H=4096
        num_stages=2, # tuned for performance
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two inputs: hidden_states [B, 4096] and weight [4096]
        hidden_states, weight = args
        # Triton path: require CUDA for performance. Provide a PyTorch fallback if tensors are not on CUDA.
        if not hidden_states.is_cuda or not weight.is_cuda:
            # Fallback uses torch ops only if CPU tensors are provided; the evaluator uses CUDA.
            x = hidden_states.to(torch.float32)
            # Note: This fallback is for non-CUDA environments; in CUDA evaluation, Triton path is used.
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)
        # Triton path: compute everything inside the kernel
        return _triton_layernorm_weight_scale(hidden_states, weight, eps=1e-5)


def run(*args):
    return ModelNew()(*args)
