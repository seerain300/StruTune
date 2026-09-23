import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,           # *bf16, (M, H)
    source_ptr,           # *bf16, (N, H)
    index_ptr,            # *int32, (N,)
    M: tl.constexpr,      # int: number of rows in output
    N,                    # int: number of source rows to process
    H: tl.constexpr,      # int: hidden size (columns)
    BLOCK_H: tl.constexpr # tile size across hidden dimension
):
    # One program per source row; distribute tiles across hidden dimension via grid(1)
    row = tl.program_id(axis=0)  # pid0
    tile = tl.program_id(axis=1) # pid1

    if row >= N:
        return

    # Destination row index for this source
    dest = tl.load(index_ptr + row)  # int32

    # Compute column offsets for this tile
    offs = tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H  # valid columns in this tile

    # Load the corresponding vector from source row
    # source_ptr is row-major: row * H + offs
    src_vals = tl.load(source_ptr + row * H + offs, mask=mask, other=0.0)

    # Atomic add to output at destination row
    # output_ptr is row-major: dest * H + offs
    tl.atomic_add(output_ptr + dest * H + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          output[row, :] = final_hidden_states[row, :] + sum over i where token_indices[i] == row of expert_outputs[i, :]
        Returns output tensor of shape (M, H).
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare output (clone-like): original code clones final_hidden_states
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        # Cast indices to int32 for Triton
        index32 = token_indices.to(torch.int32)

        output = torch.empty((M, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Choose tile size across hidden dimension; 128 works well across many GPUs
        BLOCK_H = 128

        # Grid: (N rows, ceil_div(H, BLOCK_H) tiles)
        grid = (N, triton.cdiv(H, BLOCK_H))

        # Launch kernel; num_warps can be tuned; 4 or 8 often works well
        scatter_add_rows_kernel[grid](
            output, expert_outputs, index32,
            M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )
        return output


def run(*args):
    return ModelNew()(*args)
