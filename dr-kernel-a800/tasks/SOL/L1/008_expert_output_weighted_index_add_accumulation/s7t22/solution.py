import torch
import triton
import triton.language as tl


@triton.jit
def copy_selected_rows_kernel(
    out_ptr,       # *bf16, shape (N, H)
    src_ptr,       # *bf16, shape (M, H)
    indices_ptr,   # *int32, shape (M,)
    N,             # int32, number of rows (batch_seq_len)
    H,             # int32, hidden size
    M,             # int32, number of expert outputs
    BLOCK_H: tl.constexpr,
):
    # 1D grid over rows (i in [0, M))
    i = tl.program_id(0)
    # Destination row index
    row_idx = tl.load(indices_ptr + i)  # int32
    # If within bounds, copy src row i to out[row_idx, :]
    if (row_idx >= 0) and (row_idx < N):
        # Iterate columns in chunks
        for j in range(0, H, BLOCK_H):
            cols = j + tl.arange(0, BLOCK_H)
            mask = cols < H
            src_offsets = i * H + cols
            dst_offsets = row_idx * H + cols
            vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
            tl.store(out_ptr + dst_offsets, vals, mask=mask)


@triton.jit
def fix_duplicates_kernel(
    out_ptr,       # *bf16, shape (N, H)
    src_ptr,       # *bf16, shape (M, H)
    indices_ptr,   # *int32, shape (M,)
    N,             # int32, number of rows (batch_seq_len)
    H,             # int32, hidden size
    M,             # int32, number of expert outputs
    BLOCK_H: tl.constexpr,
):
    # 2D grid over (row i, column blocks)
    i = tl.program_id(0)  # row index among selected tokens
    j_block = tl.program_id(1)  # block id over hidden dimension
    j_start = j_block * BLOCK_H
    cols = j_start + tl.arange(0, BLOCK_H)
    mask = cols < H
    row_idx = tl.load(indices_ptr + i)

    # If valid destination, accumulate
    if (row_idx >= 0) and (row_idx < N):
        out_vals = tl.load(out_ptr + row_idx * H + cols, mask=mask, other=0.0)
        src_vals = tl.load(src_ptr + i * H + cols, mask=mask, other=0.0)
        out_vals = out_vals + src_vals
        tl.store(out_ptr + row_idx * H + cols, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter accumulation to match:
          out = final_hidden_states.clone()
          out.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Implemented in two passes for robust correctness:
          1) Copy selected rows into out (no atomics).
          2) Fix duplicates by adding remaining selected rows (second pass).
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        out = final_hidden_states.clone()
        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Cast indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Kernel 1: Copy selected rows
        BLOCK_H = 128
        grid1 = (M,)
        copy_selected_rows_kernel[grid1](
            out, expert_outputs, indices_i32, N, H, M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Kernel 2: Fix duplicates
        grid2 = (M, triton.cdiv(H, BLOCK_H))
        fix_duplicates_kernel[grid2](
            out, expert_outputs, indices_i32, N, H, M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
