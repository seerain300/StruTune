import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_per_col_kernel(
    output_ptr,         # *bf16, shape (B, H)
    expert_ptr,         # *bf16, shape (T, H)
    indices_ptr,        # *i64,  shape (T,)
    B: tl.int32,        # batch_seq_len
    H: tl.int32,        # hidden_size
    T: tl.int32,        # number of expert outputs
):
    # One program per source row
    i = tl.program_id(0)
    if i >= T:
        return

    # Load target row index (convert int64 to int32 for pointer arithmetic)
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over hidden dimension, update one column at a time
    for h in range(0, H):
        # Load current value from output at (idx, h)
        curr = tl.load(output_ptr + idx * H + h)
        # Load value from expert_outputs at (i, h)
        v = tl.load(expert_ptr + i * H + h)
        # Accumulate
        new = curr + v
        # Store back
        tl.store(output_ptr + idx * H + h, new)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are on CUDA
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("All tensors must be on CUDA for Triton.")

        # Clone to match original behavior
        output = final_hidden_states.clone()
        # Make contiguous for predictable memory access
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]   # batch_seq_len
        H = output.shape[1]   # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch grid: one program per source row
        grid = (T,)

        # Run Triton kernel
        scatter_add_rows_per_col_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
