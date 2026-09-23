import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as the next power-of-two of H, clamped to [128, 1024].
    # This aims to process the entire hidden dimension in one pass for common sizes.
    if H <= 128:
        block_h = 128
    else:
        # next power-of-two
        block_h = 1 << (H - 1).bit_length()
        block_h = min(block_h, 1024)
        block_h = max(block_h, 128)

    # Heuristic for num_warps based on tile size
    if block_h <= 256:
        num_warps = 4
    elif block_h <= 512:
        num_warps = 8
    else:
        num_warps = 8  # 8 works well in practice for these tile sizes

    return block_h, num_warps


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size for H
):
    # Each program handles one source row i
    row = tl.program_id(0)

    # Bounds check for rows
    if row >= M:
        return

    # Load target token index for this row
    idx = tl.load(indices_ptr + row).to(tl.int32)

    # Column offsets for the tile
    col = tl.arange(0, BLOCK_H)

    # Process H in tiles
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + col
        mask = offs < H

        # Load the source values for this row and tile
        src_offs = row * H + offs
        vals = tl.load(src_ptr + src_offs, mask=mask, other=0.0)

        # Compute output offset and perform atomic add
        out_offs = idx * H + offs
        tl.atomic_add(out_ptr + out_offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors."
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        # Prepare output as clone of final_hidden_states
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Select BLOCK_H and num_warps based on H
        BLOCK_H, num_warps = _select_block_h_and_warps(H)

        # Launch one program per source row
        grid = (M,)  # N == M, but we map one program per row of src
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
