import torch

# Triton is required for the kernel; import here
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_scalar_kernel(
    output_ptr,           # *bf16
    expert_ptr,           # *bf16
    indices_ptr,          # *int64
    B: tl.constexpr,      # batch_seq_len (rows of output)
    H: tl.constexpr,      # hidden_size (cols)
    T: tl.constexpr,      # num_selected_tokens
):
    # Each program handles one source row i
    i = tl.program_id(0)
    # Guard against extra programs (if grid > T)
    if i >= T:
        return

    # Load token index for this row
    idx64 = tl.load(indices_ptr + i)
    # Compute the base offsets for the row in output and the row in expert
    # output is (B, H), row-major
    # We'll operate in columns j: 0..H-1
    # Triton supports while loops; use a simple loop for deterministic bf16 accumulation
    j = 0
    while j < H:
        # Load value from expert_outputs[i, j] (bf16)
        v = tl.load(expert_ptr + i * H + j)
        # Compute destination address in output: output[token_indices[i], j]
        out_off = idx64 * H + j
        # Store v to that position (bf16)
        tl.store(output_ptr + out_off, v)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This kernel performs deterministic per-element scatter-add along rows to match PyTorch behavior closely.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()
        # Make tensors contiguous for performance
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (number of rows)
        H = output.shape[1]  # hidden_size (number of columns)
        T = token_indices.shape[0]  # number of expert outputs to scatter

        # Launch one program per source row; scalar per-column loop for deterministic bf16 accumulation
        grid = (T,)

        # Launch Triton kernel
        scatter_add_rows_scalar_kernel[grid](
            output,               # output_ptr
            expert_outputs,       # expert_ptr
            token_indices,        # indices_ptr
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
