import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,  # number of rows (batch_seq_len)
    H: tl.constexpr,  # hidden_size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: rows and hidden tiles
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    # Compute row and column offsets
    row_offsets = row_block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = col_block_id * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for boundaries
    row_mask = row_offsets < B
    col_mask = col_offsets < H

    # 2D mask
    mask = row_mask[:, None] & col_mask[None, :]

    # Compute pointer offsets for 2D copy
    # Layout is row-major: dst_ptr[row*H + col]
    src_offsets = row_offsets[:, None] * H + col_offsets[None, :]
    dst_offsets = src_offsets

    # Load and store with masks
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous; Triton kernels require CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"
        final_hidden_states = final_hidden_states.contiguous()
        # Create output as a clone of final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Shapes
        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size

        # Dynamically choose tiling
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

        # Launch grid
        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        # Launch Triton copy kernel
        row_copy_kernel[grid](
            final_hidden_states,
            output,
            B,
            H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=3,
        )

        # Perform PyTorch accumulation (robust and fast)
        # Accumulate along dim=0: output[token_indices[i]] += expert_outputs[i]
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
