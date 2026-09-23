import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(
    out_ptr,       # *bf16, shape [M, H], row-major, contiguous
    src_ptr,       # *bf16, shape [M, H], row-major, contiguous
    M,             # int32: number of rows to copy
    H: tl.constexpr,  # hidden size, compile-time constant for vectorization
):
    # One program per row
    row = tl.program_id(axis=0)
    if row >= M:
        return

    offs = tl.arange(0, H)
    # Load row from source
    src_row = tl.load(src_ptr + row * H + offs)
    # Store to output
    tl.store(out_ptr + row * H + offs, src_row)


@triton.jit
def _add_rows_kernel(
    output_ptr,         # *bf16, shape [M, H], row-major, contiguous
    expert_ptr,         # *bf16, shape [tokens, H], contiguous
    indices_ptr,        # *int32, shape [tokens]
    M,                  # int32: number of rows in output
    H: tl.constexpr,    # hidden size
    tokens: tl.constexpr,  # number of programs (num_selected_tokens)
):
    # One program per token
    pid = tl.program_id(axis=0)
    if pid >= tokens:
        return

    # Load target row index for this token
    row = tl.load(indices_ptr + pid).to(tl.int32)

    # Vectorized offsets across hidden dimension
    offs = tl.arange(0, H)

    # Load existing row from output and expert row
    out_row = tl.load(output_ptr + row * H + offs)
    exp_row = tl.load(expert_ptr + pid * H + offs)

    # Elementwise add
    new_row = out_row + exp_row

    # Store back to output
    tl.store(output_ptr + row * H + offs, new_row)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        # Ensure device compatibility
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."

        # Allocate output and copy final_hidden_states into it (clone semantics)
        output = torch.empty_like(final_hidden_states)

        # Ensure inputs are contiguous for row-major addressing
        final_hidden_states = final_hidden_states.contiguous()
        output = output.contiguous()

        M = output.shape[0]
        H = output.shape[1]
        tokens = token_indices.numel()

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch copy kernel: one program per row
        grid_copy = (M,)
        _copy_rows_kernel[grid_copy](
            output,
            final_hidden_states,
            M,
            H=H,
            num_warps=4,
            num_stages=2,
        )

        # Launch add kernel: one program per token
        grid_add = (tokens,)
        _add_rows_kernel[grid_add](
            output,
            expert_outputs,
            token_indices,
            M,
            H=H,
            tokens=tokens,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
