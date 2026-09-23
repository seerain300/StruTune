import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,         # *bf16
    dst_ptr,         # *bf16
    batch_seq_len,   # int32
    hidden_size,     # int32
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Copy rows from src (final_hidden_states) to dst (output) in tiles.
    Each program handles a tile of size [BLOCK_ROWS, BLOCK_H].
    """
    pid_rows = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_rows = rows < batch_seq_len
    mask_cols = cols < hidden_size
    mask = mask_rows[:, None] & mask_cols[None, :]

    src_offsets = rows[:, None] * hidden_size + cols[None, :]
    dst_offsets = rows[:, None] * hidden_size + cols[None, :]

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized implementation:
          - Copies final_hidden_states into output using a tuned Triton kernel.
          - Performs index_add on the output using torch (exact PyTorch semantics).
        """
        # Ensure tensors are on the same CUDA device
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernel."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Dtype must be bfloat16 for these workloads."

        batch_seq_len, hidden_size = final_hidden_states.shape
        output = torch.empty_like(final_hidden_states)

        # Dynamically choose tile sizes based on hidden_size
        if hidden_size <= 128:
            BLOCK_ROWS, BLOCK_H, num_warps = 128, 128, 4
        elif hidden_size <= 512:
            BLOCK_ROWS, BLOCK_H, num_warps = 64, 256, 4
        elif hidden_size <= 1024:
            BLOCK_ROWS, BLOCK_H, num_warps = 64, 512, 8
        else:
            BLOCK_ROWS, BLOCK_H, num_warps = 32, 1024, 8

        grid = (
            triton.cdiv(batch_seq_len, BLOCK_ROWS),
            triton.cdiv(hidden_size, BLOCK_H),
        )

        row_copy_kernel[grid](
            final_hidden_states,  # src
            output,                # dst
            batch_seq_len,
            hidden_size,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Scatter-add exactly as in the original: output.index_add_(dim=0, token_indices, expert_outputs)
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
