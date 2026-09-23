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
    # 2D grid over rows and hidden dimension
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_rows = rows < B
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Compute flat offsets assuming contiguous row-major (B, H)
    src_offsets = rows[:, None] * H + cols[None, :]
    dst_offsets = rows[:, None] * H + cols[None, :]

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def triton_copy(src: torch.Tensor, dst: torch.Tensor):
    """
    Copies src -> dst using a Triton kernel. Assumes src/dst are 2D (B, H),
    contiguous, and have matching shape. Launches a 2D grid over rows and
    hidden dimension for coalesced memory access.
    """
    assert src.is_cuda and dst.is_cuda, "src and dst must be on CUDA for Triton"
    assert src.shape == dst.shape and src.ndim == 2, "src and dst must be 2D tensors of same shape"
    B, H = src.shape

    # Ensure contiguous
    src_c = src.contiguous()
    dst_c = dst.contiguous()

    # Choose block sizes based on hidden size
    if H >= 4096:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 2048:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 1024:
        BLOCK_H = 512
        num_warps = 8
    elif H >= 512:
        BLOCK_H = 256
        num_warps = 4
    else:
        BLOCK_H = 128
        num_warps = 4

    # Increase rows per program to reduce launch overhead
    BLOCK_ROWS = 128

    grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
    row_copy_kernel[grid](
        src_c, dst_c,
        B, H,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return dst_c


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Create output that starts as a clone of final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Use Triton to copy final_hidden_states -> output
        triton_copy(final_hidden_states, output)

        # Accumulate expert outputs into the correct token positions
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
