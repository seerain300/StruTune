import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_loop_kernel(
    out_ptr,           # *bfloat16, we will mutate this in-place
    expert_ptr,        # *bfloat16
    indices_ptr,       # *int64
    B: tl.int32,       # number of rows in out_ptr (batch_seq_len)
    H: tl.int32,       # number of columns (hidden_size)
    T: tl.int32,       # number of source rows
):
    # One program per source row
    i = tl.program_id(axis=0)
    if i >= T:
        return

    # Load the destination row index (int64) and cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over hidden dimension serially, add per column
    for h in range(0, H):
        # Load expert value at (i, h) as bfloat16
        val = tl.load(expert_ptr + i * H + h)
        # Load current out value at (idx, h) as bfloat16
        out_val = tl.load(out_ptr + idx * H + h)
        # Accumulate and store back
        out_val = out_val + val
        tl.store(out_ptr + idx * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Triton implementation of:
            output = final_hidden_states  # do NOT clone; mutate in-place
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the scatter-add using a Triton kernel without atomics.
        """
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Ensure contiguity for simple row-major addressing
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size
        T = token_indices.shape[0]        # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel. No clone; we mutate final_hidden_states in-place.
        scatter_add_rows_loop_kernel[grid](
            final_hidden_states,  # out_ptr: we mutate this tensor
            expert_outputs,
            token_indices,
            B, H, T,
            num_warps=1,
            num_stages=1,
        )

        return final_hidden_states


def run(*args):
    return ModelNew()(*args)
