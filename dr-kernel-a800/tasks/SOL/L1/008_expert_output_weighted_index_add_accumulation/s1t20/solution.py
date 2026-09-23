import torch
import triton
import triton.language as tl


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,           # *const bfloat16, shape [M, H]
    src_ptr,           # *const bfloat16, shape [N, H]
    indices_ptr,       # *const int32,    shape [N]
    M,                 # int32
    H,                 # int32
    BLOCK_H: tl.constexpr,
):
    # One program per source row i
    row = tl.program_id(axis=0)
    if row >= M:
        return

    # Load the destination row index for this source row
    idx = tl.load(indices_ptr + row)  # int32

    # Vectorize across hidden dimension in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load source values for this tile
        src_vals = tl.load(src_ptr + row * H + h_offsets, mask=mask, other=0.0)

        # Atomic add into output at destination row idx
        dest_ptr = out_ptr + idx * H + h_offsets
        tl.atomic_add(dest_ptr, src_vals, mask=mask)


def _select_kernel_params(H: int, N: int):
    # Choose BLOCK_H: prefer single-pass if H <= 1024; otherwise use next power-of-two clamped
    if H <= 1024:
        BLOCK_H = H
    else:
        BLOCK_H = _next_power_of_two(H)
        if BLOCK_H < 128:
            BLOCK_H = 128
        if BLOCK_H > 1024:
            BLOCK_H = 1024

    # Heuristic for num_warps and num_stages
    if BLOCK_H <= 256:
        num_warps = 4
    elif BLOCK_H <= 512:
        num_warps = 8
    else:
        num_warps = 8
    num_stages = 2
    return BLOCK_H, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        output[token_indices[i]] += expert_outputs[i] for i in [0, N),
        where output is a clone of final_hidden_states.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"

        # Clone and ensure contiguity
        out = final_hidden_states.contiguous().clone()
        src = expert_outputs.contiguous()

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = out.shape[0]
        N = src.shape[0]
        H = out.shape[1]

        # Select kernel parameters based on H and N
        BLOCK_H, num_warps, num_stages = _select_kernel_params(H, N)

        # Launch one program per source row
        grid = (N,)

        scatter_add_rows_kernel[grid](
            out, src, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
