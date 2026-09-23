import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_kernel(
    out_ptr,                    # *out (accumulator)
    expert_ptr,                 # *expert_outputs
    token_indices_ptr,          # *token_indices (int32)
    batch_seq_len,              # int
    hidden_size,                # int
    n_tokens,                   # int
    stride_out_row,             # int (out.stride(0))
    stride_out_col,             # int (out.stride(1))
    stride_exp_row,             # int (expert.stride(0))
    stride_exp_col,             # int (expert.stride(1))
    BLOCK_SIZE: tl.constexpr,   # columns processed per loop
):
    # One program per token
    token = tl.program_id(0)
    # Bounds check: if token >= n_tokens, do nothing
    if token >= n_tokens:
        return

    # Load token index (row in output where we will add the vector)
    token_index = tl.load(token_indices_ptr + token)

    # If token_index is out of range, skip (shouldn't happen if inputs are valid)
    if token_index < 0 or token_index >= batch_seq_len:
        return

    # Loop over hidden columns in chunks of BLOCK_SIZE
    col_start = 0
    while col_start < hidden_size:
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load expert vector slice: expert[token, cols]
        # Pointer arithmetic: base + token*stride_exp_row + cols*stride_exp_col
        src_ptr = expert_ptr + token * stride_exp_row + cols * stride_exp_col
        vals = tl.load(src_ptr, mask=mask, other=0.0)  # bfloat16

        # Add to output row: out[token_index, cols]
        dst_ptr = out_ptr + token_index * stride_out_row + cols * stride_out_col
        tl.store(dst_ptr, vals, mask=mask)

        col_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton-based scatter-add that matches:
          output.index_add(dim=0, index=token_indices, source=expert_outputs)
        We allocate a zero-initialized output and directly add each expert vector to the
        corresponding row using Triton kernels.
        """
        # Shapes
        batch_size = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        # Derive batch_seq_len from inputs (final_hidden_states has shape [batch_seq_len, hidden_size])
        batch_seq_len = final_hidden_states.shape[0]

        # Output accumulator: start from zeros to match index_add semantics
        out = torch.zeros((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure expert_outputs is contiguous and dtype matches
        expert_outputs = expert_outputs.contiguous()

        # Number of tokens
        n_tokens = token_indices.numel()

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose a block size; 256 is a good default for larger H, 128 for smaller
        BLOCK_SIZE = 256 if hidden_size >= 256 else 128

        # Launch scatter-add kernel: one program per token
        grid = (n_tokens,)
        scatter_add_kernel[grid](
            out, expert_outputs, token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
