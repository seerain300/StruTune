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
    # 2D launch: programs tile across rows and hidden dimension
    pid_rows = tl.program_id(0)
    pid_h = tl.program_id(1)

    row_start = pid_rows * BLOCK_ROWS
    h_start = pid_h * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)
    h = h_start + tl.arange(0, BLOCK_H)

    # 2D offsets for broadcasting
    row_offsets = rows[:, None]  # shape (BLOCK_ROWS, 1)
    h_offsets = h[None, :]       # shape (1, BLOCK_H)

    # Masks for in-bounds
    mask = (rows[:, None] < B) & (h[None, :] < H)

    # Compute flat offsets (row * H + col)
    src_offsets = row_offsets * H + h_offsets
    dst_offsets = row_offsets * H + h_offsets

    # Load and store with masks
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform the required accumulation using Triton for the copy and PyTorch for index_add.
        - final_hidden_states: [batch_seq_len, hidden_size] bfloat16
        - expert_outputs: [num_selected_tokens, hidden_size] bfloat16
        - token_indices: [num_selected_tokens] long
        Returns updated output tensor after accumulation.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA device for Triton."
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA device for Triton."
        assert token_indices.is_cuda, "token_indices must be on CUDA device for Triton."

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Prepare output tensor (clone of final_hidden_states)
        output = torch.empty_like(final_hidden_states)

        # Adaptive tiling parameters
        if H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        BLOCK_ROWS = 128  # larger tile over rows to reduce grid size and improve throughput

        grid = (
            triton.cdiv(B, BLOCK_ROWS),
            triton.cdiv(H, BLOCK_H),
        )

        # Launch Triton copy kernel
        row_copy_kernel[grid](
            final_hidden_states, output, B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Perform PyTorch index_add accumulation (fast and correct)
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
