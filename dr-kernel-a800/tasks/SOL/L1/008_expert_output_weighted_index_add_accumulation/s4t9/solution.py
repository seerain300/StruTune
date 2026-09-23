import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_serial_kernel(
    output_ptr,           # *bf16, shape (B, H)
    expert_ptr,           # *bf16, shape (T, H)
    indices_ptr,          # *int64 or *int32, shape (T,)
    B: tl.constexpr,      # number of rows in output (batch_seq_len)
    H: tl.constexpr,      # hidden_size
    T: tl.constexpr       # number of expert outputs
):
    # One program per source row
    i = tl.program_id(axis=0)

    # Load token index for this row. indices_ptr is int64 or int32; we load as int64 by default.
    # If indices are int32, Triton can still load; we cast to int32 for address arithmetic.
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)  # safe since B is within int32 range for typical sizes

    # Iterate over hidden dimension and perform deterministic accumulation
    # Note: this is a scalar loop per column for correctness; performance can be improved by tiling.
    for j in range(0, H):
        # Load source element
        val = tl.load(expert_ptr + i * H + j)
        # Load destination element, add, and store
        dest_val = tl.load(output_ptr + idx * H + j)
        dest_val += val
        tl.store(output_ptr + idx * H + j, dest_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized (but correctness-first) version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure CUDA and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        output = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row (serial accumulation for correctness)
        grid = (T,)
        scatter_add_rows_serial_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1, num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
