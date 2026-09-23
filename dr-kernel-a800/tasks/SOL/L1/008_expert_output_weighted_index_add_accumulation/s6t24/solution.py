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
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    row_start = pid_rows * BLOCK_ROWS
    col_start = pid_cols * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)
    cols = col_start + tl.arange(0, BLOCK_H)

    # 2D offsets into the tensor
    src_offsets = rows[:, None] * H + cols[None, :]
    dst_offsets = rows[:, None] * H + cols[None, :]

    # Masks for boundary
    mask_rows = rows < B
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Load from src and store to dst
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure inputs are on the same device and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA device for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Allocate output tensor (clone semantics)
        output = torch.empty((B, H), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Choose block sizes based on hidden size
        if H >= 2048:
            BLOCK_H = 1024
            BLOCK_ROWS = 128
            num_warps = 8
        elif H >= 1024:
            BLOCK_H = 512
            BLOCK_ROWS = 128
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 256
            BLOCK_ROWS = 128
            num_warps = 4
        else:
            BLOCK_H = 128
            BLOCK_ROWS = 128
            num_warps = 4

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        row_copy_kernel[grid](
            final_hidden_states, output,
            B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )

        # Perform accumulation using PyTorch (robust and correct)
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
