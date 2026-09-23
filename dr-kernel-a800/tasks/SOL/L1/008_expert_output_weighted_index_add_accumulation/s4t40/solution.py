import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,          # *bf16, shape [B, H]
    expert_ptr,          # *bf16, shape [T, H]
    indices_ptr,         # *int32, shape [T]
    B: tl.constexpr,     # batch_seq_len (rows of output)
    H: tl.constexpr,     # hidden_size (columns)
    T: tl.constexpr,     # number of expert outputs
):
    # Each program handles one source row i
    i = tl.program_id(0)
    # Guard: if grid > T, early return (safety)
    if i >= T:
        return

    # Load the destination row index for this source row
    idx = tl.load(indices_ptr + i)  # int32
    # Base pointers for this row
    out_row_base = output_ptr + idx * H
    exp_row_base = expert_ptr + i * H

    # Loop over hidden dimension and add element-wise
    for h in range(0, H):
        out_val = tl.load(out_row_base + h)          # bf16
        exp_val = tl.load(exp_row_base + h)          # bf16
        tl.store(out_row_base + h, out_val + exp_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This kernel performs scatter-add along rows (dim=0) without atomics and sequentially over columns.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 indices for addressing
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (rows to scatter into)
        H = output.shape[1]  # hidden_size (columns)
        T = token_indices_i32.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices_i32,
            B=B, H=H, T=T,
            num_warps=1, num_stages=1,
        )
        return output


def run(*args):
    return ModelNew()(*args)
