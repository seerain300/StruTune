import torch

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def zero_init_rows_cols_kernel(out_ptr, stride_row, stride_col, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # 2D grid over rows and column blocks; writes zeros to out
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row_id < n_rows) & (cols < n_cols)

    # Compute pointers for a tile (row_id, cols)
    ptrs = out_ptr + row_id * stride_row + cols * stride_col

    # Zero values (bf16)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    tl.store(ptrs, zeros, mask=mask)


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr, out_stride_row, out_stride_col,
    expert_ptr, expert_stride_row, expert_stride_col,
    token_indices_ptr,
    n_rows, n_cols, n_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token
    token_id = tl.program_id(0)

    # Bounds check for token_id in case grid > n_tokens
    if token_id >= n_tokens:
        return

    # Load token index for this token_id
    token_index = tl.load(token_indices_ptr + token_id)

    # If token_index out of bounds (shouldn't happen per get_inputs, but guard anyway)
    if (token_index < 0) | (token_index >= n_rows):
        return

    # Iterate over hidden columns in BLOCK_SIZE chunks
    start = 0
    while start < n_cols:
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Compute pointers
        out_ptrs = out_ptr + token_index * out_stride_row + cols * out_stride_col
        expert_ptrs = expert_ptr + token_id * expert_stride_row + cols * expert_stride_col

        # Load current and add
        out_vals = tl.load(out_ptrs, mask=mask, other=0.0)  # other=0 for masked lanes
        expert_vals = tl.load(expert_ptrs, mask=mask, other=0.0)

        # Atomic add into output
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)

        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Allocate output and zero-initialize (ensures atomic_add correctness)
        out = torch.zeros(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Triton requires int32 indices for simplicity; ensure contiguous
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Launch zero-init kernel: 2D grid over rows and column blocks
        BLOCK_SIZE = 128
        grid_zero = (batch_seq_len, triton.cdiv(hidden_size, BLOCK_SIZE))
        zero_init_rows_cols_kernel[grid_zero](
            out, out.stride(0), out.stride(1), batch_seq_len, hidden_size, BLOCK_SIZE, num_warps=4
        )

        # Launch scatter-add kernel: one program per token
        grid_scatter = (n_tokens,)
        scatter_add_atomic_kernel[grid_scatter](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        return out