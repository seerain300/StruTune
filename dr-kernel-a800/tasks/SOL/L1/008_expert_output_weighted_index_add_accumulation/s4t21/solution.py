import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_per_element_kernel(
    output_ptr,           # *ptr to output (B, H), dtype bfloat16
    expert_ptr,           # *ptr to expert_outputs (T, H), dtype bfloat16
    indices_ptr,          # *ptr to token_indices (T,), dtype int64
    B: tl.constexpr,      # batch_seq_len (rows of output)
    H: tl.constexpr,      # hidden_size (cols)
    T: tl.constexpr,      # number of expert outputs to scatter
):
    # Each program handles one source row i
    i = tl.program_id(axis=0)

    # Load the token index for this row i (int64 -> int32 for pointer arithmetic)
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over each hidden dimension (columns) and perform add
    for h in range(0, H):
        # Load source value: expert_outputs[i, h]
        v = tl.load(expert_ptr + i * H + h)
        # Load destination: output[idx, h]
        o = tl.load(output_ptr + idx * H + h)
        # Accumulate and store back (bf16 arithmetic)
        o = o + v
        tl.store(output_ptr + idx * H + h, o)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the scatter-add along dim=0 using a Triton kernel without atomics.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()

        # Make tensors contiguous for performance
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row; each program loops over H columns.
        grid = (T,)

        scatter_add_rows_per_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,  # simple kernel; 1 warp per program is fine
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
