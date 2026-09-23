import torch
import triton
import triton.language as tl


@triton.jit
def copy_2d_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B,        # int: number of rows (batch_seq_len)
    H,        # int: number of cols (hidden_size)
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch grid: (rows_grid, cols_grid)
    rows_block = tl.program_id(0)
    cols_block = tl.program_id(1)

    rows = rows_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = cols_block * BLOCK_H + tl.arange(0, BLOCK_H)

    # Create 2D mesh for the tile
    row_ids = rows[:, None]   # shape (BLOCK_ROWS, 1)
    col_ids = cols[None, :]   # shape (1, BLOCK_H)

    # Masks for boundaries
    mask = (row_ids < B) & (col_ids < H)

    # Compute flat offsets for 2D matrix [B, H]
    src_offsets = row_ids * H + col_ids
    dst_offsets = row_ids * H + col_ids

    # Load and store with masks
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "Triton kernels require CUDA tensors."

        # Dimensions
        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size

        # Allocate output and perform Triton copy
        output = torch.empty((B, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Choose tiling parameters based on H
        if H >= 1024:
            BLOCK_ROWS = 128
            BLOCK_H = 512
            num_warps = 8
        elif H >= 512:
            BLOCK_ROWS = 128
            BLOCK_H = 256
            num_warps = 4
        else:
            BLOCK_ROWS = 128
            BLOCK_H = 128
            num_warps = 4

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
        copy_2d_kernel[grid](
            final_hidden_states, output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )

        # Accumulate expert outputs into output at token_indices (dim=0), matching original semantics
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
