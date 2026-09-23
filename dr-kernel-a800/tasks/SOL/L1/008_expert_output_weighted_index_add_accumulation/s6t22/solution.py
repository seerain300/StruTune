import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,  # number of rows (batch_size * seq_len)
    H,                # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: pid0 over rows, pid1 over hidden tiles
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    row_offsets = pid0 * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid1 * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_rows = row_offsets < B
    mask_cols = col_offsets < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Linearized offsets for 2D tensor (B, H)
    src_offsets = row_offsets[:, None] * H + col_offsets[None, :]
    dst_offsets = src_offsets

    # Copy with masks
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def triton_copy_2d(src: torch.Tensor, dst: torch.Tensor):
    """
    Copies src -> dst using a 2D tiled Triton kernel.
    src and dst must have the same shape and be on CUDA with bfloat16 dtype.
    """
    assert src.is_cuda and dst.is_cuda, "Triton copy requires CUDA tensors"
    assert src.shape == dst.shape, "src and dst must have the same shape"
    assert src.dtype == dst.dtype, "src and dst must have the same dtype"
    B = src.shape[0]
    H = src.shape[1]

    # Choose tile sizes based on H to balance throughput and occupancy
    if H >= 2048:
        BLOCK_H = 1024
        BLOCK_ROWS = 256
        num_warps = 8
    elif H >= 1024:
        BLOCK_H = 512
        BLOCK_ROWS = 256
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
        src, dst,
        B, H,
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Allocate output and perform Triton copy
        output = torch.empty_like(final_hidden_states)
        triton_copy_2d(final_hidden_states, output)

        # Perform accumulation with PyTorch index_add (exact semantics)
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
