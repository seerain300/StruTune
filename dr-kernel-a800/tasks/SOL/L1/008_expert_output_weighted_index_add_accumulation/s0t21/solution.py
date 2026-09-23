import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_kernel(
    output_ptr,            # *bf16, shape (M, H)
    expert_ptr,            # *bf16, shape (N, H)
    indices_ptr,           # *int64, shape (N,)
    N: tl.constexpr,       # number of updates
    H: tl.constexpr,       # hidden size
    BLOCK_N: tl.constexpr, # rows per program
    BLOCK_H: tl.constexpr, # hidden features processed per iteration (can loop to cover all H)
):
    # 2D grid: axis 0 over rows (N tiles), axis 1 over H tiles
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Rows handled by this program
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Load target indices (int64)
    idxs = tl.load(indices_ptr + n_offsets, mask=n_mask, other=0).to(tl.int64)

    # Loop over H in BLOCK_H chunks to perform atomic adds
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # 2D mask for valid loads/stores
        mask_2d = n_mask[:, None] & h_mask[None, :]

        # Load expert outputs for this (BLOCK_N x BLOCK_H) tile
        expert_addrs = n_offsets[:, None] * H + h_offsets[None, :]
        vals = tl.load(expert_ptr + expert_addrs, mask=mask_2d, other=0.0)

        # Destination addresses in output: idxs[:, None] * H + h_offsets[None, :]
        dest_addrs = idxs[:, None] * H + h_offsets[None, :]

        # Atomically add vals into output
        tl.atomic_add(output_ptr + dest_addrs, vals, mask=mask_2d)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add along dim=0:
          output[token_indices[i]] += expert_outputs[i]
        where output is a clone of final_hidden_states.
        """
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA for Triton."
        device = final_hidden_states.device

        # Clone to match original semantics
        output = final_hidden_states.clone()

        # Shapes
        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs hidden size must match output's hidden size"
        assert token_indices.numel() == N, "token_indices length must equal N"

        # Choose tile sizes
        # BLOCK_N: moderate number of rows per program to increase parallelism
        if N >= 512:
            BLOCK_N = 64
        elif N >= 128:
            BLOCK_N = 32
        else:
            BLOCK_N = 16

        # BLOCK_H: process hidden dimension in chunks; for performance, use 128 or 256 when feasible
        if H >= 256:
            BLOCK_H = 256
        elif H >= 128:
            BLOCK_H = 128
        elif H >= 64:
            BLOCK_H = 64
        else:
            BLOCK_H = 32

        # Launch configuration
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        # Heuristic for num_warps: larger tiles benefit from more warps
        tile_elems = BLOCK_N * BLOCK_H
        if tile_elems >= 8192:
            num_warps = 8
        elif tile_elems >= 4096:
            num_warps = 8
        else:
            num_warps = 4

        # Launch Triton kernel
        scatter_add_2d_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
