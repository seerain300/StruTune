import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_stripe_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,  # number of rows (batch_seq_len)
    H: tl.constexpr,  # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 1D grid: each program handles a chunk of rows and a stripe along H
    pid = tl.program_id(0)

    row_start = pid * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    # For each row, we iterate H in chunks of BLOCK_H
    for col_start in range(0, H, BLOCK_H):
        cols = col_start + tl.arange(0, BLOCK_H)

        # 2D pointers for the tile
        src_ptrs = src_ptr + rows[:, None] * H + cols[None, :]
        dst_ptrs = dst_ptr + rows[:, None] * H + cols[None, :]

        # Mask for boundaries
        mask = (rows[:, None] < B) & (cols[None, :] < H)

        # Load and store (bf16). Triton will infer dtype from src_ptr.
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        tl.store(dst_ptrs, vals, mask=mask)


def triton_copy_bf16(src: torch.Tensor, dst: torch.Tensor):
    """
    Copy src -> dst using a tuned Triton kernel.
    Assumes src, dst are 2D tensors of shape (B, H) and dtype=torch.bfloat16,
    and are contiguous.
    """
    assert src.is_cuda and dst.is_cuda, "Tensors must be on CUDA."
    assert src.is_contiguous() and dst.is_contiguous(), "Tensors must be contiguous."
    assert src.dtype == torch.bfloat16 and dst.dtype == torch.bfloat16, "dtype must be bfloat16."

    B, H = src.shape

    # Choose tile sizes based on H (larger H -> larger tiles and more warps)
    if H >= 2048:
        BLOCK_H = 512
        num_warps = 8
    elif H >= 1024:
        BLOCK_H = 256
        num_warps = 8
    else:
        BLOCK_H = 128
        num_warps = 4

    # Process many rows per program to reduce grid size when B is large
    BLOCK_ROWS = 128 if B >= 1024 else 64

    grid = (triton.cdiv(B, BLOCK_ROWS),)
    copy_rows_stripe_kernel[grid](src, dst, B, H, BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H, num_warps=num_warps)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA."
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Output buffer (clone semantics)
        output = torch.empty_like(final_hidden_states)

        # Triton copy to output
        triton_copy_bf16(final_hidden_states, output)

        # Accumulate expert outputs into the correct token positions
        # Using PyTorch's highly optimized index_add along dim=0
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
