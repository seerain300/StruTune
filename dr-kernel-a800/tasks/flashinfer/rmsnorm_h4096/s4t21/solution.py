import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (float32, host provides)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.constexpr,  # hidden size (compile-time constant, e.g., 4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per iteration (e.g., 1024 or 2048)
):
    # One program per row
    row_id = tl.program_id(0)

    # Row pointers (support non-contiguous via strides)
    row_hidden = hidden_ptr + row_id * hidden_stride0
    row_out = out_ptr + row_id * out_stride0

    # 1) Compute sum of squares across the row in a single pass over H using static_range
    sumsq = 0.0
    for col in tl.static_range(0, H):
        x = tl.load(row_hidden + col * hidden_stride1)
        x = tl.cast(x, tl.float32)
        sumsq += x * x

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # 2) Compute output: y = x * inv_rms * weight[j] and store, again over H
    for col in tl.static_range(0, H):
        x = tl.load(row_hidden + col * hidden_stride1)
        x = tl.cast(x, tl.float32)
        w = tl.load(weight_ptr + col)  # weight_ptr is float32
        y = x * inv_rms * w
        tl.store(row_out + col * out_stride1, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        # Convert weight to float32 on host for math
        weight = weight.to(torch.float32).contiguous()

        B, H = hidden.shape
        assert H == 4096, "This kernel expects hidden size to be 4096"

        out = torch.empty_like(hidden)

        # Launch one program per row
        grid = (B,)

        # Use the best-performing launch parameters in this environment
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=1024,  # not directly used since we loop over H; keeps code flexible
            num_warps=8,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
