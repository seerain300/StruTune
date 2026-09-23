import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B,        # int: number of rows (batch_seq_len)
    H,        # int: hidden_size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch grid: (rows_blocks, cols_blocks)
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    rows = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    cols = pid_cols * BLOCK_H + tl.arange(0, BLOCK_H)        # [BLOCK_H]

    # Compute element offsets: each element is at (row * H + col)
    # Make 2D indexing for loads/stores
    src_offsets = rows[:, None] * H + cols[None, :]
    dst_offsets = rows[:, None] * H + cols[None, :]

    # Masks for boundary conditions
    mask = (rows[:, None] < B) & (cols[None, :] < H)

    # Load from src and store to dst
    # Using tl.load/tl.store with mask; other=0.0 is fine for bf16
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def triton_copy(final_hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Copies final_hidden_states into a new tensor using a Triton kernel.
    Returns a new tensor with the same values as final_hidden_states.
    """
    # Ensure contiguous for simple linear indexing
    src = final_hidden_states.contiguous()
    B = src.shape[0]
    H = src.shape[1]
    dst = torch.empty_like(src)

    # Choose block sizes based on shapes
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

    grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))
    # Launch kernel; use num_warps=4, num_stages=2 for balanced performance
    row_copy_kernel[grid](
        src, dst,
        B, H,
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2,
    )
    return dst


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Perform Triton copy into output
        output = triton_copy(final_hidden_states)
        # Perform PyTorch index_add for accumulation (exact semantics)
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
