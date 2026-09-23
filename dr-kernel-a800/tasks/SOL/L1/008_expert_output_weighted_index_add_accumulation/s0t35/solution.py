import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,      # *bf16/float16 pointer to output [M, H]
    expert_ptr,      # *bf16/float16 pointer to expert_outputs [N, H]
    index_ptr,       # *int32 pointer to token_indices [N]
    M,               # int: number of rows in output (batch_seq_len)
    H,               # int: number of columns (hidden_size)
    N,               # int: number of source rows (num_selected_tokens)
    BLOCK_H: tl.constexpr,  # chunk size along H
):
    # Each program handles one source row 'n'
    n = tl.program_id(axis=0)
    if n >= N:
        return

    # Destination row index for this source row
    row_index = tl.load(index_ptr + n)

    # Base pointer for this source row
    row_expert_ptr = expert_ptr + n * H

    # Vectorized offsets along H
    offs_h = tl.arange(0, BLOCK_H)

    # Iterate over H in chunks
    start = 0
    while start < H:
        h = start + offs_h
        mask = h < H

        # Load a chunk of expert outputs (coalesced along H)
        vals = tl.load(row_expert_ptr + h, mask=mask, other=0.0)

        # Compute destination addresses and perform atomic adds
        dst_ptrs = output_ptr + row_index * H + h
        tl.atomic_add(dst_ptrs, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter-add:
            final_hidden_states[row] += expert_outputs[n]
            where n in [0, num_selected_tokens), row = token_indices[n]
        Requirements:
            - final_hidden_states: (batch_seq_len, hidden_size), dtype bfloat16/float16, CUDA tensor
            - expert_outputs: (num_selected_tokens, hidden_size), dtype matches final_hidden_states
            - token_indices: (num_selected_tokens,), dtype int64/int32, CUDA tensor
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == expert_outputs.dtype, "Dtype of final_hidden_states and expert_outputs must match."
        assert final_hidden_states.dim() == 2 and expert_outputs.dim() == 2, "Inputs must be 2D."
        assert token_indices.dim() == 1, "token_indices must be 1D."

        # Ensure contiguity for coalesced loads along H
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Triton prefers int32 indices for address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Grid: one program per source row
        grid = (N,)

        # Heuristic for BLOCK_H and launch parameters (stable config that performed best)
        BLOCK_H = 128 if H >= 128 else 64
        num_warps = 4 if BLOCK_H >= 128 else 2
        num_stages = 2

        scatter_add_row_kernel[grid](
            final_hidden_states,
            expert_outputs,
            token_indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return final_hidden_states


def run(*args):
    return ModelNew()(*args)
