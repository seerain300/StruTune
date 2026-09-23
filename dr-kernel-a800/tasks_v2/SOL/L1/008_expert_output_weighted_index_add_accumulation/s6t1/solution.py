import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_atomic(
    output_ptr,         # *bf16, [batch_seq_len, hidden_size]
    expert_ptr,         # *bf16, [num_selected_tokens, hidden_size]
    indices_ptr,        # *int32, [num_selected_tokens]
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program per selected token
    token_id = tl.program_id(axis=0)

    # Load the token index (row position to accumulate into) as int32
    index = tl.load(indices_ptr + token_id)  # int32

    # Prepare offsets along the hidden dimension (vectorized lanes)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    # Load the corresponding expert_outputs row values (bf16)
    # expert_outputs is [num_selected_tokens, hidden_size], row = token_id
    expert_vals = tl.load(expert_ptr + token_id * hidden_size + offs, mask=mask, other=0.0)

    # Compute base pointer for the output row
    out_row_ptr = output_ptr + index * hidden_size

    # Atomic add into the output row at positions 'offs'
    tl.atomic_add(out_row_ptr + offs, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        # Do not modify final_hidden_states; create a new output tensor
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]

        # Allocate output tensor (start from zeros to allow atomic accumulation)
        output = torch.empty_like(final_hidden_states)

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Choose BLOCK_H; 128 is a robust default for typical hidden sizes
        BLOCK_H = 128

        # Launch Triton kernel: one program per token
        grid = (num_selected_tokens,)

        _scatter_add_atomic[grid](
            output,               # output_ptr (bf16)
            expert_outputs,       # expert_ptr (bf16)
            token_indices,        # indices_ptr (int32)
            hidden_size,          # constexpr
            BLOCK_H,              # constexpr
            num_warps=4,          # balanced default
            num_stages=2,         # modest pipelining
        )

        return output


def run(*args):
    return ModelNew()(*args)
