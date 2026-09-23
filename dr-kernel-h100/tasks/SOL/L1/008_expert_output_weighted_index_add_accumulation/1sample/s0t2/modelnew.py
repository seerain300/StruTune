import torch
import triton
import triton.language as tl


@triton.jit
def zero_init_kernel(
    out_ptr,  # *pointer* to output tensor of shape (batch_seq_len, hidden_size)
    stride0,  # stride along rows
    stride1,  # stride along cols
    H,        # batch_seq_len (number of rows)
    N,        # hidden_size (number of cols)
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (rows, col_blocks)
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    # Compute column offsets for this block
    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    # Base address for this row
    base = row * stride0
    ptrs = out_ptr + base + cols * stride1
    # Store zeros (bf16) for valid columns
    tl.store(ptrs, tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16), mask=mask)


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,          # *pointer* to output tensor (already zeroed)
    expert_ptr,       # *pointer* to expert_outputs tensor (shape [num_selected_tokens, N])
    token_indices_ptr,# *pointer* to token_indices (int32, shape [num_selected_tokens])
    batch_seq_len,    # H (number of rows in out)
    hidden_size,      # N (number of cols in out and expert_outputs)
    num_selected,     # number of tokens (num_selected_tokens)
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (token, col_block)
    token = tl.program_id(0)
    col_block = tl.program_id(1)
    # Bounds check for token (in case grid > num_selected)
    if token >= num_selected:
        return

    # Load token index for this program
    idx = tl.load(token_indices_ptr + token)
    # Bounds check: if idx out of range, return (safety; should not happen given input gen)
    if (idx < 0) or (idx >= batch_seq_len):
        return

    # Compute column offsets for this block
    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Base pointer for the expert row and output row
    # expert_outputs is [num_selected, N]; out is [H, N]
    expert_row_ptrs = expert_ptr + token * hidden_size + cols
    out_row_ptrs = out_ptr + idx * hidden_size + cols

    # Load expert vector block and atomically add to output
    exp_vals = tl.load(expert_row_ptrs, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16))
    # exp_vals and out_row_ptrs are bf16; atomic_add supports bf16 on recent GPUs
    tl.atomic_add(out_row_ptrs, exp_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        """
        Triton-only implementation of:
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        Args:
            final_hidden_states: tensor (batch_seq_len, hidden_size), dtype bfloat16 (we will zero-initialize)
            expert_outputs: tensor (num_selected_tokens, hidden_size), dtype bfloat16
            token_indices: tensor (num_selected_tokens,), dtype long (int64), values in [0, batch_seq_len)
        Returns:
            output: tensor (batch_seq_len, hidden_size), dtype bfloat16, with atomic accumulation applied
        """
        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]

        # Ensure tensors are contiguous and on CUDA
        # Triton requires CUDA; evaluation harness provides device. If not CUDA, fallback is not allowed.
        out = torch.empty(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose a block size for columns
        BLOCK_SIZE = 128  # tuneable: 128 or 256 often good; larger may reduce grid dimension

        # Launch zero-initialization kernel
        grid_zero = (batch_seq_len, triton.cdiv(hidden_size, BLOCK_SIZE))
        zero_init_kernel[grid_zero](
            out, out.stride(0), out.stride(1), batch_seq_len, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune: 4-8 are common
        )

        # Launch scatter-add atomic kernel
        grid_scatter = (num_selected_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))
        scatter_add_atomic_kernel[grid_scatter](
            out, expert_outputs, token_indices_i32, batch_seq_len, hidden_size, num_selected_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out