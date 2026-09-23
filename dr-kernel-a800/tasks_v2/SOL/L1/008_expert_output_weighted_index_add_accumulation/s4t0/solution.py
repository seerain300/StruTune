import torch
import triton
import triton.language as tl


@triton.jit
def add_by_token_index_kernel(
    output_ptr,         # *bf16, (B, H)
    expert_ptr,         # *bf16, (T, H)
    indices_ptr,        # *i32,  (T,)
    hidden_size: tl.constexpr,  # columns in the second dim (compile-time constant per launch)
):
    # One program per source row i
    i = tl.program_id(axis=0)
    # Load token index for this row
    idx = tl.load(indices_ptr + i)
    # Column offsets [0..H)
    offs = tl.arange(0, hidden_size)
    # Load the row vector from expert_outputs (row-major contiguous: i * H + offs)
    v = tl.load(expert_ptr + i * hidden_size + offs)
    # Atomic add into output row at index 'idx': output[idx, offs] += v
    tl.atomic_add(output_ptr + idx * hidden_size + offs, v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the index_add via a Triton kernel using atomic adds.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()
        # Make tensors contiguous for performance
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 for indices; convert from int64 to int32
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes
        B = final_hidden_states.shape[0]  # batch_seq_len (number of rows to scatter into)
        H = final_hidden_states.shape[1]  # hidden_size (number of columns)
        T = token_indices.shape[0]        # number of expert outputs to scatter

        # Launch one program per source row
        grid = (T,)
        num_warps = 4  # tuneable; 4 is fine for moderate H

        # Launch Triton kernel
        add_by_token_index_kernel[grid](
            output,          # output_ptr
            expert_outputs,  # expert_ptr
            token_indices,   # indices_ptr
            H,               # hidden_size as constexpr for specialization
            num_warps=num_warps,
        )

        return output


def run(*args):
    return ModelNew()(*args)
