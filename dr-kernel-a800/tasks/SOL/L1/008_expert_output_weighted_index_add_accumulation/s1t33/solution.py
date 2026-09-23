import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_block_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N] (token_indices)
    M: tl.constexpr,  # total rows in out_ptr/src_ptr
    N: tl.constexpr,  # total source rows
    H: tl.constexpr,  # hidden size
    BLOCK_R: tl.constexpr,  # number of source rows processed per program
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # 2D launch: each program handles a block of rows and a tile of H
    pid_r = tl.program_id(0)  # block id along rows
    pid_h = tl.program_id(1)  # tile id along H

    row_block_start = pid_r * BLOCK_R
    h_block_start = pid_h * BLOCK_H

    rows = row_block_start + tl.arange(0, BLOCK_R)                 # [BLOCK_R]
    h_offsets = h_block_start + tl.arange(0, BLOCK_H)             # [BLOCK_H]

    # Masks for valid indices
    mask_rows = rows < N
    mask_h = h_offsets < H

    # Load destination row indices (token positions) for each source row
    dst_rows = tl.load(indices_ptr + rows, mask=mask_rows, other=0).to(tl.int32)  # [BLOCK_R]

    # Base pointers for destination and source rows
    # Broadcast to 2D: [BLOCK_R, BLOCK_H]
    out_base = out_ptr + dst_rows[:, None] * H
    src_base = src_ptr + rows[:, None] * H

    # Build 2D mask combining row and hidden offsets
    mask_2d = mask_rows[:, None] & mask_h[None, :]

    # Load source values for the tile
    src_vals = tl.load(src_base + h_offsets[None, :], mask=mask_2d, other=0.0)  # [BLOCK_R, BLOCK_H], bfloat16

    # Atomic add into output at destination positions
    # out_base points to [dst_rows[:, None], h_offsets[None, :]]
    tl.atomic_add(out_base + h_offsets[None, :], src_vals, mask=mask_2d)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be CUDA tensors"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16"
        assert token_indices.dtype == torch.long, "token_indices must be torch.long"

        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Clone to match original semantics
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton prefers int32 indices for addressing
        indices_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: 2D grid over rows and hidden tiles
        grid = (
            triton.cdiv(N, 128),  # number of row blocks
            triton.cdiv(H, 256),  # number of hidden tiles
        )

        scatter_add_rows_block_kernel[grid](
            output, expert_outputs, indices_i32,
            M, N, H,
            BLOCK_R=128,
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
