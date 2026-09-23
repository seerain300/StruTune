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
    # 2D grid: axis 0 over rows, axis 1 over hidden blocks
    pid_rows = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Compute row and hidden indices for this program
    row_start = pid_rows * BLOCK_ROWS
    h_start = pid_h * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)
    hs = h_start + tl.arange(0, BLOCK_H)

    # Masks for boundaries
    row_mask = rows < B
    h_mask = hs < H
    mask = row_mask[:, None] & h_mask[None, :]

    # Flattened offsets within each row
    src_offsets = rows[:, None] * H + hs[None, :]
    dst_offsets = rows[:, None] * H + hs[None, :]

    # Load from source and store to destination
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        final_hidden_states: (batch_seq_len, hidden_size), bfloat16, device
        expert_outputs: (num_selected_tokens, hidden_size), bfloat16, device
        token_indices: (num_selected_tokens,), long, device
        Returns: output tensor with contributions added at token_indices.
        """
        # Ensure inputs are on the same device and contiguous
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Output buffer: start with a clone of final_hidden_states using Triton
        output = torch.empty_like(final_hidden_states)

        # Choose BLOCK_H and BLOCK_ROWS based on shape
        if H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        BLOCK_ROWS = 128

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
        row_copy_kernel[grid](
            final_hidden_states,
            output,
            B,
            H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Accumulate expert_outputs into output at token_indices using PyTorch's optimized index_add
        # Equivalent to: for i in range(num_selected_tokens): output[token_indices[i]] += expert_outputs[i]
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
