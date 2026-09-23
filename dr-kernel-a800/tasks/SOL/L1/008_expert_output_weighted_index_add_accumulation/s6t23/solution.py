import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B,        # int: number of rows (batch_seq_len)
    H,        # int: hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: each program handles a tile of rows and hidden columns
    pid_rows = tl.program_id(0)
    pid_h = tl.program_id(1)

    row_offsets = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)            # [BLOCK_H]

    # Compute masks for boundaries
    row_mask = row_offsets < B
    h_mask = h_offsets < H
    mask = row_mask[:, None] & h_mask[None, :]

    # Compute pointers for the tile
    src_tile_ptr = src_ptr + row_offsets[:, None] * H + h_offsets[None, :]
    dst_tile_ptr = dst_ptr + row_offsets[:, None] * H + h_offsets[None, :]

    # Load and store with masks
    vals = tl.load(src_tile_ptr, mask=mask, other=0.0)
    tl.store(dst_tile_ptr, vals, mask=mask)


def triton_row_copy(final_hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Copies final_hidden_states to a new output tensor using a Triton 2D copy kernel.
    Returns the copied tensor.
    """
    assert final_hidden_states.is_cuda, "Input must be on CUDA for Triton kernel."
    B, H = final_hidden_states.shape
    output = torch.empty_like(final_hidden_states)

    # Choose tile sizes based on hidden size
    if H >= 2048:
        BLOCK_H, BLOCK_ROWS, num_warps = 1024, 128, 8
    elif H >= 1024:
        BLOCK_H, BLOCK_ROWS, num_warps = 512, 128, 8
    elif H >= 512:
        BLOCK_H, BLOCK_ROWS, num_warps = 256, 128, 4
    else:
        BLOCK_H, BLOCK_ROWS, num_warps = 128, 128, 4

    grid = (
        triton.cdiv(B, BLOCK_ROWS),
        triton.cdiv(H, BLOCK_H),
    )

    row_copy_kernel[grid](
        final_hidden_states, output,
        B, H,
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors."

        # Triton-optimized copy: output starts as clone of final_hidden_states
        output = triton_row_copy(final_hidden_states)

        # Robust, optimized accumulation with PyTorch
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
