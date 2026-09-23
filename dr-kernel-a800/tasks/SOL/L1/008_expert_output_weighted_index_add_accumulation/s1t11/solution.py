import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_vec_kernel(
    out_ptr,            # *bf16, shape (M, H)
    expert_ptr,         # *bf16, shape (N, H)
    idx_ptr,            # *int32, shape (N,)
    N,                  # int32 runtime
    H,                  # int32 runtime
    BLOCK_H: tl.constexpr,
):
    # One program per source row
    row = tl.program_id(0)
    if row >= N:
        return

    # Load token index for this row
    idx = tl.load(idx_ptr + row).to(tl.int32)

    # Iterate over hidden dimension in tiles
    for j in range(0, H, BLOCK_H):
        offs = j + tl.arange(0, BLOCK_H)  # shape: (BLOCK_H,)
        mask = offs < H

        # Compute base pointers
        # Output row pointer: (M, H) row selected by idx
        out_row_ptr = out_ptr + idx * H + offs  # broadcastable (BLOCK_H,)
        # Expert row pointer: (N, H) row for the current source row
        expert_row_ptr = expert_ptr + row * H + offs

        # Load expert outputs for this tile
        val = tl.load(expert_row_ptr, mask=mask, other=0.0)

        # Atomic add into the output row at those hidden positions
        tl.atomic_add(out_row_ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        """
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"

        # Clone to match original semantics
        out = final_hidden_states.clone()

        # Triton prefers int32 for indexing
        idx32 = token_indices.to(torch.int32).contiguous()
        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()

        # Shapes
        M, H = out.shape
        N, H_exp = expert_outputs.shape
        assert H_exp == H, "expert_outputs hidden size must match output hidden size"

        # Tile size across hidden dimension
        BLOCK_H = 128  # good default for H up to 1024; adjust if needed

        # Launch one program per source row
        grid = (N,)

        _scatter_add_rows_vec_kernel[grid](
            out, expert_outputs, idx32,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
