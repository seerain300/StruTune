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
    # 2D grid over rows and hidden-dim tiles
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    rows = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]  # shape (BLOCK_ROWS, 1)
    cols = pid_cols * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]        # shape (1, BLOCK_H)

    mask = (rows < B) & (cols < H)

    # Compute flat offsets: row-major for 2D tensor (B, H)
    src_offsets = rows * H + cols
    dst_offsets = src_offsets

    # Load from source and store to destination
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def triton_copy(final_hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Copy final_hidden_states to a new tensor using Triton.
    Keeps dtype and device; returns a new tensor with the same contents.
    """
    assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA for Triton kernel."
    B, H = final_hidden_states.shape
    output = torch.empty_like(final_hidden_states)

    # Choose tile sizes based on hidden size
    if H >= 1024:
        BLOCK_H = 512
        num_warps = 4
    elif H >= 512:
        BLOCK_H = 256
        num_warps = 4
    else:
        BLOCK_H = 128
        num_warps = 4

    # Choose tile along rows based on B
    if B >= 8192 or B >= 4096:
        BLOCK_ROWS = 256
    elif B >= 1024:
        BLOCK_ROWS = 128
    else:
        BLOCK_ROWS = 64

    grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

    row_copy_kernel[grid](
        final_hidden_states, output,
        B, H,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton copy to produce output that starts with final_hidden_states
        output = triton_copy(final_hidden_states)

        # Accumulate expert outputs into selected token positions (dim=0)
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
