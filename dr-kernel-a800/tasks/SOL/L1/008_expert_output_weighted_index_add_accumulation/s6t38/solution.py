import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,   # *const bfloat16
    dst_ptr,   # *bfloat16
    B: tl.constexpr,  # int: number of rows (batch_seq_len)
    H: tl.constexpr,  # int: hidden size
    BLOCK_ROWS: tl.constexpr,  # tile size along rows
    BLOCK_H: tl.constexpr,     # tile size along hidden dimension
):
    # 2D launch: pid0 over rows, pid1 over hidden tiles
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    row_start = pid0 * BLOCK_ROWS
    col_start = pid1 * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)           # shape [BLOCK_ROWS]
    cols = col_start + tl.arange(0, BLOCK_H)              # shape [BLOCK_H]

    # Create 2D pointer grid for a tile
    src_offsets = rows[:, None] * H + cols[None, :]       # [BLOCK_ROWS, BLOCK_H]
    dst_offsets = rows[:, None] * H + cols[None, :]       # [BLOCK_ROWS, BLOCK_H]

    # Bounds masks
    mask_rows = rows < B
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Load and store with masking
    src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity for simpler addressing
        if not final_hidden_states.is_contiguous():
            final_hidden_states = final_hidden_states.contiguous()
        # Allocate output and perform Triton copy
        output = torch.empty_like(final_hidden_states)

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Choose tile sizes based on shapes
        if H >= 2048:
            BLOCK_H = 1024
        elif H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        if B >= 8192:
            BLOCK_ROWS = 128
        elif B >= 2048:
            BLOCK_ROWS = 64
        else:
            BLOCK_ROWS = 32

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        # Launch Triton kernel
        row_copy_kernel[grid](
            final_hidden_states, output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=8, num_stages=3
        )

        # Perform PyTorch accumulation (robust and fast)
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
