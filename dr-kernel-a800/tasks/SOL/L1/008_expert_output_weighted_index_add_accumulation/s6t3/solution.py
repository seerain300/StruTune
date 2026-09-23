import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(
    out_ptr,           # *bf16, shape [M, H], row-major
    src_ptr,           # *bf16, shape [M, H], row-major
    M: tl.constexpr,   # batch_seq_len (rows)
    H: tl.constexpr,   # hidden_size (cols)
    BLOCK: tl.constexpr,  # tile width over columns
):
    # Each program handles one row
    row_id = tl.program_id(0)
    # Column offsets for the tile
    col_offsets = tl.arange(0, BLOCK)
    # Loop over columns in tiles of size BLOCK
    for col_start in range(0, H, BLOCK):
        cols = col_start + col_offsets
        # Compute flattened indices for load/store (row-major)
        src_idx = row_id * H + cols
        out_idx = row_id * H + cols
        vals = tl.load(src_ptr + src_idx)
        tl.store(out_ptr + out_idx, vals)


@triton.jit
def scatter_copy_add_kernel(
    out_ptr,            # *bf16, shape [M, H], row-major
    expert_ptr,         # *bf16, shape [N, H]
    indices_ptr,        # *int32, shape [N]
    M: tl.constexpr,    # batch_seq_len (rows)
    N: tl.constexpr,    # num_selected_tokens
    H: tl.constexpr,    # hidden_size (cols)
):
    # 2D launch: (row_id, tok_id)
    row_id = tl.program_id(0)
    tok_id = tl.program_id(1)

    # Bounds masks
    row_mask = row_id < M
    tok_mask = tok_id < N

    # Load token index for this tok_id
    idx = tl.load(indices_ptr + tok_id, mask=tok_mask, other=0)

    # Check if this token corresponds to the current row
    is_match = (idx == row_id) & tok_mask

    # Column offsets (vectorized across hidden dimension)
    col_offsets = tl.arange(0, H)

    # Compute pointers
    out_row_ptrs = out_ptr + row_id * H + col_offsets
    expert_row_ptrs = expert_ptr + tok_id * H + col_offsets

    # Load existing row from output and expert row
    out_vals = tl.load(out_row_ptrs, mask=row_mask, other=0.0)
    expert_vals = tl.load(expert_row_ptrs, mask=tok_mask, other=0.0)

    # Only update when idx == row_id; otherwise leave output unchanged
    new_vals = tl.where(is_match, out_vals + expert_vals, out_vals)

    # Store back
    tl.store(out_row_ptrs, new_vals, mask=row_mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized replacement for the PyTorch index_add accumulation.
        - Copies final_hidden_states into output first (clone semantics).
        - Then performs scatter-add: for each token i, if token_indices[i] == r, add expert_outputs[i] to output[r, :].
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Output buffer: start from empty, then we fill via copy + scatter-add
        output = torch.empty_like(final_hidden_states)

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch copy kernel: one program per row
        BLOCK = 128  # tile size for columns; 128 is a good default
        grid_copy = (M,)
        copy_rows_kernel[grid_copy](
            output, final_hidden_states, M, H, BLOCK,
            num_warps=4, num_stages=2
        )

        # Launch scatter-add kernel: grid over rows and tokens
        grid_add = (M, N)
        scatter_copy_add_kernel[grid_add](
            output, expert_outputs, token_indices, M, N, H,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
