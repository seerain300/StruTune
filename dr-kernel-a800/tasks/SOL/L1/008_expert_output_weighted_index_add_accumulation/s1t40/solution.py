import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H to minimize iterations and maintain good occupancy.
    # For typical H <= 1024, using 1024 reduces to a single pass.
    if H <= 1024:
        block_h = 1024
    elif H <= 2048:
        block_h = 512
    else:
        block_h = 256
    # Warps heuristic: more warps for larger tiles.
    if block_h >= 1024:
        num_warps = 8
    elif block_h >= 512:
        num_warps = 8
    else:
        num_warps = 4
    num_stages = 2
    return block_h, num_warps, num_stages


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M: tl.constexpr,  # total rows (unused but can be used for bounds checking if desired)
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # total number of source rows
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row i
    row = tl.program_id(axis=0)
    # If grid > N, mask out extra programs
    if row >= N:
        return

    # Load token index for this row
    idx = tl.load(indices_ptr + row)
    # Compute base pointers for this row
    out_row_ptr = out_ptr + idx * H
    src_row_ptr = src_ptr + row * H

    # Vector of column offsets within a tile
    offs_h = tl.arange(0, BLOCK_H)

    # Iterate over H in tiles
    # In practice, for H <= 1024 and BLOCK_H=1024, this loop runs once.
    for start in range(0, H, BLOCK_H):
        h = start + offs_h
        mask = h < H
        # Load source vector (bf16), masked for tail
        src_vec = tl.load(src_row_ptr + h, mask=mask, other=0.0)
        # Atomic add into output at positions [idx, h]
        tl.atomic_add(out_row_ptr + h, src_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
          output = final_hidden_states.clone()
          for i in range(N): output[token_indices[i]] += expert_outputs[i]
        """
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA for Triton kernel"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA for Triton kernel"
        assert token_indices.is_cuda, "token_indices must be on CUDA for Triton kernel"

        # Ensure contiguous and dtypes
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # PyTorch index_add expects indices as long; Triton kernel uses int32
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Output buffer: clone from input (device-side)
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        out = torch.empty_like(final_hidden_states)

        # Select tile size and warps based on H
        BLOCK_H, num_warps, num_stages = _select_block_h_and_warps(H)

        # Launch one program per source row
        grid = (triton.cdiv(N, 1),)  # grid size equals N
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
