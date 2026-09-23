import torch
import triton
import triton.language as tl


@triton.jit
def zero_init_kernel(out_ptr, stride_row, stride_col, batch_len, hidden_size, BLOCK_SIZE: tl.constexpr):
    # 2D grid: (rows, col_blocks)
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Compute linear offsets: out[row_id, cols]
    offs = row_id * stride_row + cols * stride_col
    # Write zeros
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    tl.store(out_ptr + offs, zeros, mask=mask)


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr, stride_row, stride_col,
    expert_ptr, expert_stride_row, expert_stride_col,
    token_indices_ptr,
    batch_len, hidden_size, n_tokens, BLOCK_SIZE: tl.constexpr
):
    # Grid: (tokens, col_blocks)
    token_id = tl.program_id(0)  # which token we process
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < hidden_size

    # Load token index for this program
    token_index = tl.load(token_indices_ptr + token_id)  # int32

    # Compute base offsets
    # out: out[token_index, cols]
    out_offs = token_index * stride_row + cols * stride_col
    # expert: expert[token_id, cols]
    expert_offs = token_id * expert_stride_row + cols * expert_stride_col

    # Load current out values and expert contributions (bfloat16)
    out_vals = tl.load(out_ptr + out_offs, mask=col_mask, other=0.0)
    expert_vals = tl.load(expert_ptr + expert_offs, mask=col_mask, other=0.0)

    # Accumulate
    partial = out_vals + expert_vals

    # Atomic add into output
    tl.atomic_add(out_ptr + out_offs, partial, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Compute dimensions
        batch_size = final_hidden_states.shape[0]
        seq_len = final_hidden_states.shape[1]  # hidden_size is the second dimension
        hidden_size = final_hidden_states.shape[1]
        batch_seq_len = batch_size * seq_len

        # Allocate output (clone semantics of original: zero-init then add)
        out = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Zero-initialize output using Triton
        BLOCK_SIZE_ZERO = 256
        grid_zero = (batch_seq_len, triton.cdiv(hidden_size, BLOCK_SIZE_ZERO))
        zero_init_kernel[grid_zero](
            out, out.stride(0), out.stride(1), batch_seq_len, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE_ZERO,
            num_warps=4,
        )

        # Launch scatter-add kernel: 2D grid over tokens and column blocks
        n_tokens = expert_outputs.shape[0]
        BLOCK_SIZE = 256
        grid_scatter = (n_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))
        scatter_add_atomic_kernel[grid_scatter](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
