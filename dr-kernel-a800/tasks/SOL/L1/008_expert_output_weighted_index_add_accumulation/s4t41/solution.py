import torch
import triton
import triton.language as tl


@triton.jit
def add_row_vector_kernel(
    output_ptr,          # *bf16, shape (B, H)
    expert_ptr,          # *bf16, shape (T, H)
    indices_ptr,         # *int64, shape (T,)
    H: tl.constexpr,     # hidden_size (number of columns)
):
    # One program per source row i
    i = tl.program_id(0)
    # Load destination row index
    idx64 = tl.load(indices_ptr + i)  # int64
    idx = idx64.to(tl.int32)          # cast to int32 for pointer arithmetic

    # Vector of column offsets
    offs = tl.arange(0, H)

    # Load the source row vector (bf16)
    v = tl.load(expert_ptr + i * H + offs)  # shape (H,), dtype bfloat16

    # Compute output offsets for the selected row
    out_offsets = idx * H + offs

    # Add in-place: output[idx, :] += v
    out_vals = tl.load(output_ptr + out_offsets)
    out_vals = out_vals + v
    tl.store(output_ptr + out_offsets, out_vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement per-row vector addition using Triton to match PyTorch's index_add semantics exactly.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()
        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (number of rows)
        H = output.shape[1]  # hidden_size (number of columns)
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel: add each expert row to the corresponding output row
        add_row_vector_kernel[grid](
            output, expert_outputs, token_indices,
            H=H,
            num_warps=4,   # tuneable
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
