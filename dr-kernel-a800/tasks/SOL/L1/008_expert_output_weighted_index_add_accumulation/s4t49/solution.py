import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_loop_kernel(
    output_ptr,        # *bf16, pointer to output tensor [B, H]
    expert_ptr,        # *bf16, pointer to expert_outputs tensor [T, H]
    indices_ptr,       # *int64, pointer to token_indices tensor [T]
    B,                 # int32, number of rows in output (batch_seq_len)
    H,                 # int32, number of columns (hidden_size)
    T,                 # int32, number of expert outputs / tokens
):
    # Each program handles one source row i
    i = tl.program_id(0)
    if i >= T:
        return

    # Load the destination row index (int64 -> int32)
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over hidden dimension H one column at a time
    # We use a Python-level loop; Triton will unroll or schedule appropriately.
    for h in range(0, H):
        # Load v from expert_outputs[i, h] (bf16)
        v = tl.load(expert_ptr + i * H + h)  # pointer arithmetic: row-major
        # Store v into output[idx, h]
        tl.store(output_ptr + idx * H + h, v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure all tensors are on CUDA for Triton execution
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (rows)
        H = output.shape[1]  # hidden_size (cols)
        T = token_indices.shape[0]  # number of expert outputs to scatter

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel
        scatter_add_rows_loop_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
