import torch
import triton
import triton.language as tl


@triton.jit
def _multiply_hidden_by_index_kernel(
    hidden_ptr,  # *const bfloat16
    out_ptr,     # *bfloat16
    num_tokens: tl.constexpr,  # int
    H: tl.constexpr,            # hidden size (compile-time constant)
):
    t = tl.program_id(0)  # one program per token
    if t >= num_tokens:
        return

    # base pointers for this token
    row_hidden_ptr = hidden_ptr + t * H
    row_out_ptr = out_ptr + t * H

    # process each column j = 0..H-1
    for j in range(0, H):
        x = tl.load(row_hidden_ptr + j)  # bfloat16
        # y = x * (j + 1), keeping dtype bfloat16
        y = x * tl.cast(j + 1, x.dtype)
        tl.store(row_out_ptr + j, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [num_tokens, hidden_size], dtype bfloat16
        assert hidden_states.is_cuda, "Input must be on CUDA device."
        assert hidden_states.dtype == torch.bfloat16, "Input dtype must be bfloat16."
        num_tokens, H = hidden_states.shape
        out = torch.empty_like(hidden_states)

        # Launch one program per token
        grid = (num_tokens,)
        _multiply_hidden_by_index_kernel[grid](
            hidden_states, out,
            num_tokens=num_tokens,
            H=H,
            num_warps=1,  # small problem size; 1 warp is sufficient
        )
        return out


def run(*args):
    return ModelNew()(*args)
