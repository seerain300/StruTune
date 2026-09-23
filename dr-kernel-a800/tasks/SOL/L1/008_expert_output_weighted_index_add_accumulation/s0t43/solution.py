import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,     # *bf16, shape [M, H], contiguous
    expert_ptr,     # *bf16, shape [N, H], contiguous
    index_ptr,      # *int32, shape [N]
    M: tl.int32,    # total rows in output
    H: tl.int32,    # hidden size (columns)
    N: tl.int32,    # number of source rows to add
    BLOCK_H: tl.constexpr,  # chunk size along H
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load token index for this source row (int32)
    token = tl.load(index_ptr + i)  # int32

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)          # [BLOCK_H], int32
        mask = h_offsets < H                                # valid columns mask

        # Compute addresses (use int64 for safety)
        # output[token, h_offsets]
        dest_row_offset = token.to(tl.int64) * tl.full((), H, tl.int64)
        dest_col_offset = h_offsets.to(tl.int64)
        dest_addr = output_ptr + (dest_row_offset + dest_col_offset)  # [BLOCK_H], int64

        # expert[i, h_offsets]
        src_row_offset = i.to(tl.int64) * tl.full((), H, tl.int64)
        src_addr = expert_ptr + (src_row_offset + h_offsets.to(tl.int64))  # [BLOCK_H], int64

        # Load values with mask and atomically add
        vals = tl.load(src_addr, mask=mask, other=0.0)      # [BLOCK_H] bf16
        tl.atomic_add(dest_addr, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure contiguity and dtypes
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Make token_indices int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = output.shape[0]
        H = output.shape[1]
        N = token_indices.shape[0]

        # Fixed, stable configuration that performed well across tests
        # Use slightly larger BLOCK_H for very large H to reduce loop iterations
        if H >= 1024:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2

        # Launch one program per source row
        grid = (N,)
        scatter_add_row_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            M,
            H,
            N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
