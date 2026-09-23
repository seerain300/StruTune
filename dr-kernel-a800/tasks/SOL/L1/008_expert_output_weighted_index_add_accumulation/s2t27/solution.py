import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,         # *bf16, pointer to (M, H) output tensor
    in_ptr,          # *bf16, pointer to (N, H) expert_outputs tensor
    idx_ptr,         # *int32, pointer to (N,) token_indices
    M,               # int, number of rows in output (batch_seq_len)
    N,               # int, number of selected tokens (num_selected_tokens)
    H,               # int, hidden size
    BLOCK: tl.constexpr,  # chunk size along hidden dimension
):
    # One program per selected token (row)
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Base pointer for the source row
    in_row_ptr = in_ptr + pid * H

    # Column offsets for a chunk
    col_offsets = tl.arange(0, BLOCK)

    # Number of chunks to cover H
    num_chunks = (H + BLOCK - 1) // BLOCK

    # Iterate over hidden dimension in BLOCK-sized chunks
    for chunk in range(0, num_chunks):
        start = chunk * BLOCK
        cols = start + col_offsets
        mask = cols < H

        # Load source values for this chunk (bf16)
        vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0)  # shape (BLOCK,), bfloat16

        # Read destination row index
        dest_row = tl.load(idx_ptr + pid).to(tl.int32)

        # Compute destination row base pointer
        out_row_ptr = out_ptr + dest_row * H + start + col_offsets

        # Atomic add the chunk into output
        tl.atomic_add(out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA and have expected dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expect bfloat16 tensors"

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = output.shape[0]  # batch_seq_len
        N = expert_outputs.shape[0]  # num_selected_tokens
        H = output.shape[1]  # hidden_size

        # Launch one program per selected token
        grid = (N,)

        # Robust, performant configuration (no autotune to avoid intermittent issues)
        BLOCK = 128
        _scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, N, H,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
