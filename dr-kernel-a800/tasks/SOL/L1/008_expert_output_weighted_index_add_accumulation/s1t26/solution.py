import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    N,                # number of source rows to process
    H,                # hidden size
    BLOCK_H: tl.constexpr,
):
    # One Triton program handles one source row
    row = tl.program_id(0)
    if row >= N:
        return

    # Destination row index for this source row
    dest_idx = tl.load(indices_ptr + row)  # int32

    # Column tile offsets
    offs = tl.arange(0, BLOCK_H)

    # Iterate over hidden dimension in tiles
    for col_start in range(0, H, BLOCK_H):
        cols = col_start + offs
        mask = cols < H

        # Load source row slice (bfloat16)
        src_row_ptr = src_ptr + row * H + cols
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Atomic add to destination row slice
        out_row_ptr = out_ptr + dest_idx * H + cols
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


def _select_block_h_and_warps(H: int):
    # Next power-of-two of H, clamped to [128, 1024]
    if H <= 128:
        block_h = 128
        num_warps = 4
    else:
        block_h = 1 << ((H - 1).bit_length())
        block_h = min(max(block_h, 128), 1024)
        if block_h <= 256:
            num_warps = 4
        else:
            num_warps = 8
    return block_h, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton forward requires CUDA tensors."
        # Expect bfloat16 tensors
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "This kernel expects bfloat16 tensors."

        # Shapes
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Make inputs contiguous
        out = final_hidden_states.contiguous()  # output buffer
        src = expert_outputs.contiguous()      # source rows

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Select tile size and warps
        BLOCK_H, num_warps = _select_block_h_and_warps(H)

        # Launch kernel: one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, src, token_indices,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
