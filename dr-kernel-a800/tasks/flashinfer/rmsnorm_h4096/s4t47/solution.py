import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16)
    weight_ptr,       # *const weight tensor (e.g., bfloat16/float16)
    out_ptr,          # *output tensor
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (expect 4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    # One Triton program per row
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Row pointers
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1
    out_row_ptr = out_ptr + row_id * out_stride0 + cols * out_stride1

    # Load hidden row and cast to float32 for math
    x = tl.load(hidden_row_ptr, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)

    # Compute y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result (cast to out_ptr's dtype automatically)
    tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # If not CUDA, provide a PyTorch fallback (evaluation uses CUDA, so kernel will be used)
        if not (hidden_states.is_cuda and weight.is_cuda):
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguity
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "This Triton kernel expects hidden size 4096"

        # Allocate output tensor with same shape and dtype as input
        out = torch.empty_like(hidden)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, w, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8,  # tuned for H=4096
            num_stages=2, # tuned for H=4096
        )
        return out


def run(*args):
    return ModelNew()(*args)
