import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,              # *bf16
    expert_ptr,           # *bf16
    token_indices_ptr,    # *int32
    batch_seq_len,        # int32
    hidden_size,          # int32
    n_tokens,             # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token
    i = tl.program_id(0)
    # Guard in case grid > n_tokens (not used here, but safe)
    if i >= n_tokens:
        return

    # Load token index (row destination)
    token_index = tl.load(token_indices_ptr + i)

    # Loop over columns in chunks of BLOCK_SIZE
    for col in range(0, hidden_size, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Compute base offsets
        # out_ptr is laid out row-major: offset = token_index * stride_row + cols * stride_col
        # But since we set out to be contiguous (2D), we can use linear indexing:
        # out_row_ptr = out_ptr + token_index * hidden_size
        out_row_ptr = out_ptr + token_index * hidden_size
        out_vec = tl.load(out_row_ptr + cols, mask=mask, other=0.0)

        # Load corresponding expert vector chunk
        expert_row_ptr = expert_ptr + i * hidden_size
        exp_vec = tl.load(expert_row_ptr + cols, mask=mask, other=0.0)

        # Accumulate in-place in the kernel (per-token duplication handled correctly)
        out_vec = out_vec + exp_vec

        # Store back
        tl.store(out_row_ptr + cols, out_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        where output has shape (batch_seq_len, hidden_size), expert_outputs has shape (num_selected_tokens, hidden_size),
        and token_indices has shape (num_selected_tokens,) with values in [0, batch_seq_len).
        """
        # Ensure tensors are on CUDA and have correct dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton"
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Allocate output and zero-initialize (torch does it efficiently and correctly)
        out = torch.empty(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Triton kernel expects int32 indices
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Launch one program per token
        BLOCK_SIZE = 128  # tuneable; 128 or 256 often good
        grid = (n_tokens,)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
