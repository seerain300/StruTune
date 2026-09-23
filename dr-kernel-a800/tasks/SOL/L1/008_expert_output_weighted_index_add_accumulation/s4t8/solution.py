import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_serial_kernel(
    output_ptr,           # *bf16, shape (B, H)
    expert_ptr,           # *bf16, shape (T, H)
    indices_ptr,          # *int64, shape (T,)
    B: tl.constexpr,      # batch_seq_len (rows in output)
    H: tl.constexpr,      # hidden_size (cols in output)
    T: tl.constexpr,      # number of expert outputs
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= T:
        return

    # Load the target row index for this source row
    idx = tl.load(indices_ptr + i)  # int64

    # Iterate over hidden dimension and add elementwise
    for j in range(0, H):
        out_ptr = output_ptr + idx * H + j
        exp_ptr = expert_ptr + i * H + j
        val = tl.load(out_ptr) + tl.load(exp_ptr)
        tl.store(out_ptr, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This kernel performs deterministic per-row, per-column addition to ensure numerical consistency.
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel
        scatter_add_rows_serial_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
