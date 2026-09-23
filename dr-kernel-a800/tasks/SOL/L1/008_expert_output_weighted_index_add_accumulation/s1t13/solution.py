import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,     shape [N]
    M,                # int32: number of rows in output (batch_size * seq_len)
    H,                # int32: hidden size
    N,                # int32: number of expert outputs to add
    BLOCK_H: tl.constexpr,  # tile size along hidden dimension
):
    # One program per source row
    row = tl.program_id(0)

    # Iterate over H in tiles
    j = 0
    while j < H:
        offsets = j + tl.arange(0, BLOCK_H)
        mask = offsets < H

        # Load the source row slice (contiguous along H)
        src_row_ptr = src_ptr + row * H
        val = tl.load(src_row_ptr + offsets, mask=mask, other=0.0)  # bfloat16

        # Load the token index for this row (int32)
        idx = tl.load(indices_ptr + row)

        # Atomic add into the output row at the selected position
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offsets, val, mask=mask)

        j += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        where:
          - final_hidden_states: [M, H], bfloat16
          - expert_outputs:       [N, H], bfloat16
          - token_indices:        [N],    int64 (will be cast to int32)
        Returns: [M, H], bfloat16
        """
        # Ensure contiguity
        out = final_hidden_states.contiguous().clone()
        src = expert_outputs.contiguous()
        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = out.shape[0]
        H = out.shape[1]
        N = src.shape[0]

        # Launch one program per row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, src, token_indices,
            M, H, N,
            BLOCK_H=256,      # 1 iteration for H <= 1024
            num_warps=8,      # higher parallelism for vectorized ops
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
