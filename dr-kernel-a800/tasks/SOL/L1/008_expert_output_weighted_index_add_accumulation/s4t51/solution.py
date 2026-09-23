import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_per_element_kernel(
    output_ptr,           # *bf16, shape (B, H)
    expert_ptr,           # *bf16, shape (T, H)
    indices_ptr,          # *int64, shape (T,)
    B: tl.constexpr,      # batch_seq_len (number of rows in output)
    H: tl.constexpr,      # hidden_size (number of columns)
    T: tl.constexpr,      # number of expert outputs to scatter
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= T:
        return

    # Load destination row index (int64), cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate over hidden dimension and perform per-element scatter
    for h in range(0, H):
        # Load value from expert_outputs[i, h] (row-major: i*H + h)
        val = tl.load(expert_ptr + i * H + h)
        # Store into output[idx, h]
        tl.store(output_ptr + idx * H + h, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the scatter-add by launching one Triton program per source row i,
        and writing each element expert_outputs[i, h] into output[token_indices[i], h].
        """
        # Ensure tensors are on the same CUDA device
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == expert_outputs.dtype, "final_hidden_states and expert_outputs must have the same dtype"

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs to scatter

        # Launch one program per source row
        grid = (T,)

        # Launch Triton kernel
        scatter_add_rows_per_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,  # small kernel, 1 warp is sufficient
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
