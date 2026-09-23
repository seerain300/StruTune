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
    # 2D grid: tile across rows and hidden dimension
    pid_rows = tl.program_id(0)
    pid_h = tl.program_id(1)

    row_start = pid_rows * BLOCK_ROWS
    h_start = pid_h * BLOCK_H

    # Tile coordinates: rows x hs
    rows = row_start + tl.arange(0, BLOCK_ROWS)[:, None]  # shape (BLOCK_ROWS, 1)
    hs = h_start + tl.arange(0, BLOCK_H)[None, :]        # shape (1, BLOCK_H)

    # Mask for valid elements
    mask = (rows < B) & (hs < H)

    # Linear offsets (row-major contiguous: stride(0) = H, stride(1) = 1)
    offsets = rows * H + hs

    # Load from source and store to destination
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Allocate output and perform Triton copy
        output = torch.empty_like(final_hidden_states)
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Choose tiling based on hidden size
        if H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        BLOCK_ROWS = 128
        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        row_copy_kernel[grid](
            final_hidden_states, output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=2,
        )

        # Perform index-add accumulation in PyTorch (fast and robust)
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
