import torch
import triton
import triton.language as tl


@triton.jit
def zero_init_kernel(
    out_ptr,
    out_stride_row, out_stride_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid over rows and column blocks
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Compute column indices for this block
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for in-bounds columns
    mask = cols < n_cols

    # Compute linear offsets for the (row, cols) positions
    offsets = row * out_stride_row + cols * out_stride_col

    # Write zeros to these positions
    vals = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def scatter_add_rows_kernel(
    out_ptr, expert_ptr, token_idx_ptr,
    out_stride_row, out_stride_col,
    exp_stride_row, exp_stride_col,
    n_rows, n_cols, n_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: one program per token and per column block
    token = tl.program_id(0)
    col_block = tl.program_id(1)

    # Load token index (row destination)
    tok = tl.load(token_idx_ptr + token)
    # Guard: if token >= n_rows, skip (defensive; token_indices should be valid)
    if tok >= n_rows:
        return

    # Compute column indices for this block
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Compute offsets for out row
    out_offsets = tok * out_stride_row + cols * out_stride_col
    # Compute offsets for expert row
    exp_offsets = token * exp_stride_row + cols * exp_stride_col

    # Load expert values for this token's columns
    vals = tl.load(expert_ptr + exp_offsets, mask=mask, other=0.0)

    # Store into output
    tl.store(out_ptr + out_offsets, vals, mask=mask)


def _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices):
    """
    Triton-based scatter-add along rows:
    out[token_indices[i]] += expert_outputs[i] for all i.

    - final_hidden_states: (batch_seq_len, hidden_size) bfloat16
    - expert_outputs: (num_selected_tokens, hidden_size) bfloat16
    - token_indices: (num_selected_tokens,) int64 (converted to int32 for Triton)
    Returns: updated out tensor with the accumulated contributions.
    """
    # Ensure tensors are on CUDA and contiguous
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
    batch_seq_len = final_hidden_states.shape[0]
    hidden_size = final_hidden_states.shape[1]
    n_tokens = expert_outputs.shape[0]

    # Zero-initialize the output (robust zeroing)
    out = torch.zeros(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

    # Triton expects int32 indices
    token_indices_i32 = token_indices.to(torch.int32).contiguous()

    # Choose block size for columns
    BLOCK_SIZE = 128  # tuneable: 128 or 256 are typical

    # Launch zero-init kernel: 2D grid over rows and column blocks
    num_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    grid_zero = (batch_seq_len, num_blocks)
    zero_init_kernel[grid_zero](
        out, out.stride(0), out.stride(1),
        batch_seq_len, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    # Launch scatter-add kernel: 2D grid over tokens and column blocks
    grid_scatter = (n_tokens, num_blocks)
    scatter_add_rows_kernel[grid_scatter](
        out, expert_outputs, token_indices_i32,
        out.stride(0), out.stride(1),
        expert_outputs.stride(0), expert_outputs.stride(1),
        batch_seq_len, hidden_size, n_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Run Triton scatter-add (ensure tensors are on CUDA device)
        if not final_hidden_states.is_cuda:
            final_hidden_states = final_hidden_states.cuda(non_blocking=True)
        if not expert_outputs.is_cuda:
            expert_outputs = expert_outputs.cuda(non_blocking=True)
        if not token_indices.is_cuda:
            token_indices = token_indices.cuda(non_blocking=True)
        return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)