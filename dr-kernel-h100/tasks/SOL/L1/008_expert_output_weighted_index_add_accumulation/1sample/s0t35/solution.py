import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,                      # *bf16
    out_stride0, out_stride1,     # int64 strides for out
    exp_ptr,                      # *bf16
    exp_stride0, exp_stride1,     # int64 strides for exp
    token_indices,                # *int32
    batch_seq_len, hidden_size, n_tokens,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token and one hidden block
    pid_tok = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Compute column offsets for this block
    col_start = pid_col * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < hidden_size

    # Load token index for this program
    tok_idx = tl.load(token_indices + pid_tok)
    # Optional: mask invalid tokens if any (usually not needed as indices are valid)
    # If you want to guard, uncomment:
    # valid = tok_idx >= 0 and tok_idx < batch_seq_len
    # (Triton doesn't support Python 'and' here; but indices are generated in [0, batch_seq_len).)

    # Compute pointers for out and exp slices
    out_row_ptr = out_ptr + tok_idx * out_stride0
    exp_row_ptr = exp_ptr + pid_tok * exp_stride0

    # Pointers for this column block
    out_ptrs = out_row_ptr + col_offsets * out_stride1
    exp_ptrs = exp_row_ptr + col_offsets * exp_stride1

    # Load exp slice with mask, then atomic_add into out
    exp_vals = tl.load(exp_ptrs, mask=col_mask, other=0.0)
    tl.atomic_add(out_ptrs, exp_vals, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform: output = final_hidden_states.clone(); then output.index_add_(dim=0, token_indices, expert_outputs).
        Triton kernel performs scatter-add with atomic accumulation.
        """
        # Preserve initial random values
        output = final_hidden_states.clone()

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Triton expects indices as int32
        token_indices_i32 = token_indices.to(torch.int32)

        # Ensure contiguous expert tensor
        expert_outputs = expert_outputs.contiguous()

        # Choose tile size and warps based on hidden_size
        if hidden_size <= 512:
            BLOCK_SIZE = 512
            num_warps = 8
        else:
            BLOCK_SIZE = 256
            num_warps = 4

        # 2D grid over tokens and hidden column blocks
        grid = (n_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))

        # Launch scatter-add kernel
        scatter_add_atomic_kernel[grid](
            output, output.stride(0), output.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
