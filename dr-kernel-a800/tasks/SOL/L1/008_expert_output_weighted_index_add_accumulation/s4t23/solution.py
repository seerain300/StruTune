import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_serial_kernel(
    output_ptr,          # *bf16
    expert_ptr,          # *bf16
    indices_ptr,         # *int64
    B: tl.constexpr,     # batch_seq_len (rows of output)
    H: tl.constexpr,     # hidden_size (cols of output)
    T: tl.constexpr,     # number of source rows (token_indices length)
):
    # One program per source row i
    i = tl.program_id(0)
    # Bounds check
    if i >= T:
        return

    # Load token index as int64 then cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over hidden dimension serially to minimize numerical differences
    for h in range(0, H):
        # Load the value from expert_outputs[i, h] as bfloat16
        val = tl.load(expert_ptr + i * H + h)  # bf16 scalar
        # Load the corresponding output row element
        out_val = tl.load(output_ptr + idx * H + h)  # bf16 scalar
        # Add and store back
        out_val = out_val + val
        tl.store(output_ptr + idx * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the scatter-add via a Triton kernel that iterates over the hidden dimension
        serially for each source row to minimize numerical discrepancies.
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity (row-major)
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel; use small num_warps/num_stages for numerical stability
        scatter_add_rows_serial_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
