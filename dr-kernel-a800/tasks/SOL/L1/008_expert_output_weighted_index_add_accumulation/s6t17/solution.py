import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B,        # int: number of rows (batch_seq_len)
    H,        # int: number of hidden columns
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: (rows_tiles, cols_tiles)
    rows_tile = tl.program_id(0)
    cols_tile = tl.program_id(1)

    # Compute row and column offsets for this program
    row_offsets = rows_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = cols_tile * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for bounds
    row_mask = row_offsets < B
    col_mask = col_offsets < H
    mask = row_mask[:, None] & col_mask[None, :]

    # Compute pointers for source and destination tiles
    src_ptrs = src_ptr + row_offsets[:, None] * H + col_offsets[None, :]
    dst_ptrs = dst_ptr + row_offsets[:, None] * H + col_offsets[None, :]

    # Load and store (bf16) with masks
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized clone (copy) followed by PyTorch index_add accumulation.
        - final_hidden_states: [batch_seq_len, hidden_size], bfloat16, device
        - expert_outputs: [num_selected_tokens, hidden_size], bfloat16, device
        - token_indices: [num_selected_tokens], int64 (long), values in [0, batch_seq_len)
        Returns output of shape [batch_seq_len, hidden_size] with semantics:
          output = final_hidden_states.clone()
          output.index_add_(0, token_indices, expert_outputs)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtype must be bfloat16."
        assert token_indices.dtype == torch.long, "token_indices must be torch.long (int64)."

        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]

        # Allocate output and copy via Triton
        output = torch.empty((B, H), device=final_hidden_states.device, dtype=final_hidden_states.dtype)

        # Choose block sizes based on hidden size
        if H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        BLOCK_ROWS = 64

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
        row_copy_kernel[grid](
            final_hidden_states, output, B, H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=8,   # higher warps for better copy throughput on GPUs
            num_stages=3,  # more stages to hide latency
        )

        # Perform accumulation with PyTorch (guaranteed correctness, robust to duplicates)
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
