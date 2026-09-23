import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.int32,   # number of rows (batch_seq_len)
    H: tl.int32,   # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D tiling over rows and hidden dimension
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    row_start = row_block_id * BLOCK_ROWS
    col_start = col_block_id * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)
    cols = col_start + tl.arange(0, BLOCK_H)

    mask_rows = rows < B
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Compute source and destination offsets: row * H + col
    src_offsets = rows[:, None] * H + cols[None, :]
    dst_offsets = rows[:, None] * H + cols[None, :]

    # Load and store with masking; masked elements use 0.0
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous for Triton
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Prepare output as a clone of final_hidden_states
        output = torch.empty((B, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Choose tile sizes based on shape
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

        row_copy_kernel[grid](
            final_hidden_states, output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=8, num_stages=3,
        )

        # Accumulate expert outputs back to token positions using PyTorch (highly optimized)
        # This operation is: output[token_indices[i]] += expert_outputs[i]
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
