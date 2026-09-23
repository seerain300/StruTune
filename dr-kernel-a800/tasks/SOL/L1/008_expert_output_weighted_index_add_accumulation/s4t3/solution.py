import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,      # *bfloat16, shape [B, H]
    expert_ptr,      # *bfloat16, shape [T, H]
    indices_ptr,     # *int64,    shape [T]
    B,               # runtime int: number of rows in output (batch_size * seq_len)
    H: tl.constexpr, # compile-time int: number of columns in output (hidden_size)
    T,               # runtime int: number of source rows
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= T:
        return

    # Load destination row index (int64), cast to int32 for address arithmetic
    idx64 = tl.load(indices_ptr + i)  # int64
    idx = idx64.to(tl.int32)

    # Process all hidden columns deterministically
    for h in range(0, H):
        # Load expert value for this row and column
        val = tl.load(expert_ptr + i * H + h)  # bfloat16
        # Load current output value at destination row and column
        curr = tl.load(output_ptr + idx * H + h)  # bfloat16
        # Accumulate and store back
        new = curr + val
        tl.store(output_ptr + idx * H + h, new)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We deterministically perform the per-element scatter-add along rows (dim=0)
        without atomics to ensure exact correctness.
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
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
