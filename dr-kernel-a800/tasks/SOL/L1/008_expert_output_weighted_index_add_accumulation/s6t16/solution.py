import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    n_rows: tl.constexpr,  # int
    n_cols: tl.constexpr,  # int
    stride_src_row, stride_src_col,
    stride_dst_row, stride_dst_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: each program handles a tile of rows x hidden columns
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    row_start = pid_rows * BLOCK_ROWS
    col_start = pid_cols * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    cols = col_start + tl.arange(0, BLOCK_H)     # [BLOCK_H]

    # 2D tile indices
    row_ids = rows[:, None]  # [BLOCK_ROWS, 1]
    col_ids = cols[None, :]  # [1, BLOCK_H]

    # Bounds masks
    mask_rows = row_ids < n_rows
    mask_cols = col_ids < n_cols
    mask = mask_rows & mask_cols  # [BLOCK_ROWS, BLOCK_H]

    # Compute offsets using strides (row-major)
    src_offsets = row_ids * stride_src_row + col_ids * stride_src_col
    dst_offsets = row_ids * stride_dst_row + col_ids * stride_dst_col

    # Load and store
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version:
          1) Copy final_hidden_states to output using a Triton 2D-tiled kernel.
          2) torch.index_add along dim=0 to add expert_outputs at token_indices.
        """
        # Ensure tensors are on CUDA and bfloat16 as per provided inputs
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA for Triton kernel."
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA for Triton kernel."
        assert token_indices.is_cuda, "token_indices must be on CUDA for Triton kernel."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "This implementation expects bfloat16 tensors."

        n_rows = final_hidden_states.shape[0]
        n_cols = final_hidden_states.shape[1]
        output = torch.empty_like(final_hidden_states)

        # Ensure contiguous for simple stride-based addressing
        src = final_hidden_states.contiguous()
        dst = output  # will be written by kernel

        # Choose tile sizes based on hidden size
        H = n_cols
        if H >= 1024:
            BLOCK_H = 512
            num_warps = 4
        elif H >= 512:
            BLOCK_H = 256
            num_warps = 4
        else:
            BLOCK_H = 128
            num_warps = 4

        BLOCK_ROWS = 64

        grid = (
            triton.cdiv(n_rows, BLOCK_ROWS),
            triton.cdiv(H, BLOCK_H),
        )

        row_copy_kernel[grid](
            src, dst,
            n_rows, H,
            src.stride(0), src.stride(1),
            dst.stride(0), dst.stride(1),
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        # Perform scatter-add exactly as in the reference
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
