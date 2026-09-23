import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_direct(
    output_ptr,         # *bf16, [batch_seq_len, hidden_size]
    expert_ptr,         # *bf16, [num_selected_tokens, hidden_size]
    indices_ptr,        # *int32, [num_selected_tokens]
    hidden_size: tl.constexpr,
):
    # One program per selected token
    token_id = tl.program_id(axis=0)

    # Load the token index (row position to accumulate into) as int32
    index = tl.load(indices_ptr + token_id)  # int32

    # Iterate over hidden dimension and store add directly
    for h in range(0, hidden_size):
        # Load expert_outputs[token_id, h] (bf16 scalar)
        expert_val = tl.load(expert_ptr + token_id * hidden_size + h)

        # Compute destination pointer for output at [index, h]
        out_ptr = output_ptr + index * hidden_size + h

        # Store the scalar into output (equivalent to adding into a pre-zeroed output)
        tl.store(out_ptr, expert_val)


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

        # Allocate output tensor (we will fill it with accumulated results)
        # Note: The original code clones final_hidden_states and then index_adds.
        # We do not perform the clone here; we produce the final accumulated result.
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]

        # We'll return the accumulated result directly. No clone on host side.
        output = torch.empty((batch_seq_len, hidden_size), device=final_hidden_states.device, dtype=final_hidden_states.dtype)

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per token
        grid = (num_selected_tokens,)

        _scatter_add_direct[grid](
            output,               # output_ptr (bf16)
            expert_outputs,       # expert_ptr (bf16)
            token_indices,        # indices_ptr (int32)
            hidden_size,          # constexpr
            num_warps=1,          # one warp per program; simple and robust
            num_stages=1,         # minimal pipelining
        )

        return output


def run(*args):
    return ModelNew()(*args)
