import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_element_kernel(
    output_ptr,          # *bf16, shape (B, H)
    expert_ptr,          # *bf16, shape (T, H)
    indices_ptr,         # *int64, shape (T,)
    B: tl.constexpr,     # number of rows in output (batch_seq_len)
    H: tl.constexpr,     # hidden size (columns)
    T: tl.constexpr      # number of source rows (num_selected_tokens)
):
    # One program per source row i
    i = tl.program_id(0)  # i in [0, T)
    if i >= T:
        return

    # Load token index as int64
    idx64 = tl.load(indices_ptr + i)
    # We will use int64 for pointer arithmetic to avoid any int32 range issues.
    # Triton pointer arithmetic supports int64.

    # Iterate over hidden columns h = 0..H-1
    for h in range(0, H):
        # Load value from expert_outputs[i, h] (row-major: i * H + h)
        v = tl.load(expert_ptr + i * H + h)  # bf16 load

        # Store into output[ idx, h ]
        tl.store(output_ptr + (idx64 * H + h), v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the scatter-add via a Triton kernel without atomics, using per-element stores.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity for pointer arithmetic
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel: per-element scatter-add
        scatter_add_rows_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
