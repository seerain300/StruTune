import torch
import triton
import triton.language as tl


def _next_power_of_two(x: int) -> int:
    # Returns the next power of two >= x, with a minimum of 128 and a maximum of 1024
    if x <= 128:
        return 128
    # Next power of two
    p = 1 << (x - 1).bit_length()
    return min(max(p, 128), 1024)


@triton.jit
def initialize_zeros_kernel(
    out_ptr,  # *bfloat16, shape [M, H]
    M,        # int
    H: tl.constexpr,
):
    # Simple kernel: fill the entire output tensor with zeros
    # We'll launch a 2D grid (rows, cols) to parallelize over the matrix.
    row = tl.program_id(0)
    col = tl.program_id(1)
    if row < M and col < H:
        ptr = out_ptr + row * H + col
        tl.store(ptr, 0.0)


@triton.jit
def scatter_add_tiles_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    N,                # int
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # 2D grid: [i in [0, N), tile_id in [0, ceil_div(H, BLOCK_H))]
    i = tl.program_id(0)
    tile_id = tl.program_id(1)
    # If i >= N (shouldn't happen with grid setup), return
    if i >= N:
        return

    # Compute offsets for this hidden tile
    h0 = tile_id * BLOCK_H
    h_offsets = h0 + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    # Destination row index for this source i
    dst_row = indices_ptr[i]

    # Compute pointers for this tile
    out_ptrs = out_ptr + dst_row * H + h_offsets
    src_ptrs = src_ptr + i * H + h_offsets

    # Load source values (masked for tail), then atomic add into output
    val = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.atomic_add(out_ptrs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and dtypes match expectations
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."

        # output: clone of final_hidden_states -> initialize to zeros
        M, H = final_hidden_states.shape
        output = torch.empty_like(final_hidden_states)  # this allocates and can be zero-initialized via Triton kernel
        # Cast indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)
        N = expert_outputs.shape[0]

        # Determine BLOCK_H and warps
        BLOCK_H = _next_power_of_two(H)
        # Heuristic for warps: 4 for small tiles, 8 for larger
        num_warps = 4 if BLOCK_H <= 256 else 8

        # Launch zero-initialization kernel (2D grid over MxH)
        grid_init = (M, H)
        initialize_zeros_kernel[grid_init](
            output,
            M,
            H=H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Launch scatter-add kernel with 2D grid over rows and hidden tiles
        grid_scatter = (N, (H + BLOCK_H - 1) // BLOCK_H)
        scatter_add_tiles_kernel[grid_scatter](
            output, expert_outputs, indices_i32,
            N,
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
