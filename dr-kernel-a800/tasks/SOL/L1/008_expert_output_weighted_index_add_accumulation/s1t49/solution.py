import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in output
    N: tl.constexpr,  # total source rows (M * num_experts_per_tok)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size across hidden dimension
):
    # Each program handles one source row (i in [0, N))
    src_row = tl.program_id(0)

    # Base pointer for the current source row
    src_row_ptr = src_ptr + src_row * H

    # Loop over the hidden dimension in tiles
    for col_start in range(0, H, BLOCK_H):
        offs = col_start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load source tile
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Load destination row index
        dest_row = tl.load(indices_ptr + src_row)

        # Compute output pointer for this tile
        out_row_ptr = out_ptr + dest_row * H
        out_tile_ptr = out_row_ptr + offs

        # Atomic add into output
        tl.atomic_add(out_tile_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Triton-optimized scatter-add:
          output = final_hidden_states.clone()
          for i in [0, N): output[token_indices[i]] += expert_outputs[i]
        Assumes token_indices are in [0, M) where M = final_hidden_states.shape[0].
        """

        # Ensure dtypes and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 tensors."
        assert token_indices.dtype == torch.int32, "token_indices must be int32."

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape [N, H]."
        assert token_indices.shape == (N,), "token_indices must have shape [N]."

        # Prepare output as clone of final_hidden_states
        output = final_hidden_states.clone()

        # Launch Triton kernel: one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            M,
            N,
            H,
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
