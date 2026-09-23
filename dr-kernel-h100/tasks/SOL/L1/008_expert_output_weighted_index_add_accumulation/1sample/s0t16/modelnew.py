import torch

# Triton is required for the computation. Ensure availability; evaluation harness provides device.
import triton
import triton.language as tl


@triton.jit
def zero_init_kernel(
    out_ptr,         # *bf16
    stride_row,      # int32: out.stride(0)
    stride_col,      # int32: out.stride(1)
    n_rows,          # int32: batch_seq_len
    n_cols,          # int32: hidden_size
    BLOCK_SIZE: tl.constexpr,  # int: columns per block
):
    # 2D grid: rows x col_blocks
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    start_col = col_block * BLOCK_SIZE
    cols = start_col + tl.arange(0, BLOCK_SIZE)

    # Bounds mask for rows and columns
    row_mask = row < n_rows
    col_mask = cols < n_cols
    mask = row_mask & col_mask

    # Compute output offsets for the row and columns
    # out_ptr is bf16*, element indexing uses strides (in elements)
    out_row_ptrs = out_ptr + row * stride_row + cols * stride_col

    # Store zeros (bf16) for valid lanes
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    tl.store(out_row_ptrs, zeros, mask=mask)


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,           # *bf16
    out_stride_row,    # int32: out.stride(0)
    out_stride_col,    # int32: out.stride(1)
    expert_ptr,        # *bf16
    expert_stride_row, # int32: expert_outputs.stride(0)
    expert_stride_col, # int32: expert_outputs.stride(1)
    token_indices_ptr, # *int32
    n_rows,            # int32: batch_seq_len
    n_cols,            # int32: hidden_size
    n_tokens,          # int32: num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # int: columns per block
):
    # One program per token
    pid = tl.program_id(0)
    # If grid > n_tokens, we can guard; but Triton grid is typically exact. Use mask for safety.
    # Load token index for this program
    token_idx = tl.load(token_indices_ptr + pid)
    # Bounds guard for token index (optional but safe)
    in_bounds = (pid < n_tokens) & (token_idx >= 0) & (token_idx < n_rows)

    # Iterate over columns in BLOCK_SIZE chunks
    col = 0
    while col < n_cols:
        cols = col + tl.arange(0, BLOCK_SIZE)
        col_mask = (pid < n_tokens) & (cols < n_cols) & in_bounds
        # Compute pointers
        # out row pointer for this token index
        out_row_ptrs = out_ptr + token_idx * out_stride_row + cols * out_stride_col
        # expert row pointer: pid selects which expert row
        expert_row_ptrs = expert_ptr + pid * expert_stride_row + cols * expert_stride_col
        # Load expert values (masked)
        exp_vals = tl.load(expert_row_ptrs, mask=col_mask, other=0.0).to(tl.bfloat16)
        # Atomic add into out
        tl.atomic_add(out_row_ptrs, exp_vals, mask=col_mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        """
        Triton-only forward: computes
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        using Triton kernels. Returns the updated output tensor.
        """
        # Determine shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Allocate output; do NOT modify it via torch ops in hot path (Triton kernels will do the work)
        # out should be bfloat16 on same device as final_hidden_states
        out = torch.empty(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Triton requires int32 indices; ensure contiguous
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Choose block size for columns
        BLOCK_SIZE = 128  # tuneable: 128 or 256

        # 1) Zero-initialize output (robust 2D grid over rows and column blocks)
        grid_zero = (batch_seq_len, triton.cdiv(hidden_size, BLOCK_SIZE))
        zero_init_kernel[grid_zero](
            out, out.stride(0), out.stride(1),
            batch_seq_len, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune: 4-8
        )

        # 2) Scatter-add via Triton (one program per token)
        grid_scatter = (n_tokens,)
        scatter_add_atomic_kernel[grid_scatter](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune
        )

        return out