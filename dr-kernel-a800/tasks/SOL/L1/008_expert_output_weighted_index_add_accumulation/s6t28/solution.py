import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,  # number of rows (batch_seq_len)
    H: tl.constexpr,  # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid over rows and hidden dimension tiles
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    row_offsets = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_cols * BLOCK_H + tl.arange(0, BLOCK_H)

    # Compute pointers for the tile
    src_tile_ptr = src_ptr + row_offsets[:, None] * H + col_offsets[None, :]
    dst_tile_ptr = dst_ptr + row_offsets[:, None] * H + col_offsets[None, :]

    # Masks for boundaries
    row_mask = row_offsets < B
    col_mask = col_offsets < H
    mask = row_mask[:, None] & col_mask[None, :]

    # Load and store; for bf16 tensors, default is fine
    vals = tl.load(src_tile_ptr, mask=mask, other=0.0)
    tl.store(dst_tile_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        final_hidden_states: (batch_seq_len, hidden_size), bfloat16
        expert_outputs: (num_selected_tokens, hidden_size), bfloat16
        token_indices: (num_selected_tokens,), int64
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Allocate output and copy using Triton for performance
        output = torch.empty_like(final_hidden_states)
        # Choose block sizes based on H
        BLOCK_H = 256 if H >= 512 else 128
        BLOCK_ROWS = 256
        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
        row_copy_kernel[grid](
            final_hidden_states,
            output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # Accumulate with PyTorch (fast and correct for duplicates)
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
