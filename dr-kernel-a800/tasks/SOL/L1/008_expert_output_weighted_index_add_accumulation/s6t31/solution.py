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
    # 2D launch grid over rows and hidden dimension tiles
    pid_rows = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute row and hidden offsets for this program
    row_offsets = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for boundaries
    mask_rows = row_offsets < B
    mask_cols = col_offsets < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Compute linear offsets for src and dst (contiguous [B, H])
    src_offsets = row_offsets[:, None] * H + col_offsets[None, :]
    dst_offsets = src_offsets  # same layout

    # Load and store with masks
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton copy for final_hidden_states to output, then PyTorch index_add for accumulation.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtype must be bfloat16"
        assert final_hidden_states.is_contiguous() and expert_outputs.is_contiguous(), "Tensors must be contiguous"

        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size

        # Output buffer: copy final_hidden_states into it
        output = torch.empty_like(final_hidden_states)

        # Choose block sizes based on H
        if H >= 1024:
            BLOCK_H = 512
            num_warps = 4
        elif H >= 512:
            BLOCK_H = 256
            num_warps = 4
        else:
            BLOCK_H = 128
            num_warps = 4

        # Use BLOCK_ROWS = 256 to reduce launch overhead on larger B
        BLOCK_ROWS = 256

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        row_copy_kernel[grid](
            final_hidden_states, output,
            B=B, H=H,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Accumulate expert_outputs into the correct token positions
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
