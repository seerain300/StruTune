import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr, stride_row, stride_col,
    expert_ptr, stride_exp_row, stride_exp_col,
    token_indices_ptr,
    batch_seq_len, hidden_size, n_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token
    token_id = tl.program_id(0)
    # Bound check: if token_id >= n_tokens, do nothing (grid size should equal n_tokens)
    # Triton will not launch extra programs if grid > n_tokens, so this is fine.
    # Load destination row index for this token
    row = tl.load(token_indices_ptr + token_id)

    # Iterate over hidden_size in chunks of BLOCK_SIZE
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load expert vector for this token at these columns
        expert_vec = tl.load(
            expert_ptr + token_id * stride_exp_row + cols * stride_exp_col,
            mask=mask, other=0.0
        )

        # Atomic add the expert contributions into output at (row, cols)
        tl.atomic_add(
            out_ptr + row * stride_row + cols * stride_col,
            expert_vec,
            mask=mask
        )


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Output should preserve the initial random values, then accumulate expert contributions
        output = final_hidden_states.clone()

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Triton expects indices as int32
        token_indices_i32 = token_indices.to(torch.int32)

        # Launch kernel: one program per token
        BLOCK_SIZE = 128  # tuneable; 128 or 256 are good defaults
        grid = (n_tokens,)
        scatter_add_atomic_kernel[grid](
            output, output.stride(0), output.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
