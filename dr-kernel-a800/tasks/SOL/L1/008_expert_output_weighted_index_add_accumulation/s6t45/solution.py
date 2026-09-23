import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,   # number of rows (batch_seq_len)
    H: tl.constexpr,   # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: program_id(0) over rows, program_id(1) over hidden columns
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    row_offsets = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_cols * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_rows = row_offsets < B
    mask_cols = col_offsets < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # src and dst are row-major; each row has H elements contiguous
    src_row_ptrs = src_ptr + row_offsets[:, None] * H
    dst_row_ptrs = dst_ptr + row_offsets[:, None] * H

    src_ptrs = src_row_ptrs + col_offsets[None, :]
    dst_ptrs = dst_row_ptrs + col_offsets[None, :]

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure dtype/device compatibility
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype == torch.long, "token_indices must be torch.long"

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Output starts as a clone of final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Choose tiling based on hidden_size
        if H >= 2048:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 1024:
            BLOCK_H = 512
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 256
            num_warps = 8
        else:
            BLOCK_H = 128
            num_warps = 4

        # Larger BLOCK_ROWS to reduce grid size along rows
        BLOCK_ROWS = 128

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        # Launch Triton copy kernel
        row_copy_kernel[grid](
            final_hidden_states,  # src
            output,                # dst
            B, H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=3,
        )

        # Perform accumulation using PyTorch (robust and efficient)
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
