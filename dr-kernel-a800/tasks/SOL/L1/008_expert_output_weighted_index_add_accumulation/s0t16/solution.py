import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_kernel(
    output_ptr,        # *bfloat16, shape (M, H)
    expert_ptr,        # *bfloat16, shape (N, H)
    indices_ptr,       # *int32,    shape (N,)
    M: tl.constexpr,   # number of rows in output (batch_seq_len)
    H: tl.constexpr,   # hidden size
    N: tl.constexpr,   # number of updates
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: axis 0 over N in tiles of BLOCK_N, axis 1 over H in tiles of BLOCK_H
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Rows this program handles
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Hidden features this program handles
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Load token indices for these rows (int32), and cast to int64 for address arithmetic
    # Shape: (BLOCK_N,)
    idxs = tl.load(indices_ptr + n_offsets, mask=mask_n, other=0)
    idxs = idxs.to(tl.int64)  # target row in output for each source row

    # Compute destination and source pointers for 2D tile
    # Destination: output[idxs[n], h]
    # Source:      expert[n, h]
    # We'll broadcast to (BLOCK_N, BLOCK_H) for vectorized operations.
    # Output pointers for a tile:
    # out_base = idxs * H + h_offsets
    # out_ptrs = output_ptr + out_base
    # src_base = n_offsets * H + h_offsets
    # src_ptrs = expert_ptr + src_base

    # Broadcast to 2D
    # idxs: (BLOCK_N,), h_offsets: (BLOCK_H,)
    # Create meshgrid
    n_idx = n_offsets[:, None]           # (BLOCK_N, 1)
    h_vec = h_offsets[None, :]           # (1, BLOCK_H)
    idxs_broadcast = idxs[:, None]       # (BLOCK_N, 1)
    n_offsets_broadcast = n_offsets[:, None]  # (BLOCK_N, 1)
    h_broadcast = h_offsets[None, :]     # (1, BLOCK_H)

    # Masks for 2D
    mask_2d = (mask_n[:, None] & mask_h[None, :])

    # Compute element indices for output and expert
    out_indices = idxs_broadcast * H + h_broadcast
    src_indices = n_offsets_broadcast * H + h_broadcast

    # Pointer arithmetic
    out_ptrs = output_ptr + out_indices
    src_ptrs = expert_ptr + src_indices

    # Load source tile (bfloat16)
    src_vals = tl.load(src_ptrs, mask=mask_2d, other=0.0)

    # Atomic add into output
    tl.atomic_add(out_ptrs, src_vals, mask=mask_2d)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          output = final_hidden_states.clone()
          output.index_add(dim=0, index=token_indices, source=expert_outputs)
        All computation is performed by Triton kernel(s).
        """
        # Clone input buffer to initialize output (required for correctness)
        # This is the only tensor operation allowed in forward (host-side).
        output = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        device = final_hidden_states.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA device"
        assert expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be CUDA tensors"

        M, H = output.shape
        N = expert_outputs.shape[0]

        # Triton prefers int32 indices when possible; token_indices are in [0, M), so int32 is safe.
        # If token_indices is int64, cast to int32 (value-preserving).
        if token_indices.dtype != torch.int32:
            token_indices_32 = token_indices.to(torch.int32)
        else:
            token_indices_32 = token_indices

        # Choose tiling parameters
        # Heuristic: larger H -> larger BLOCK_H; larger N -> larger BLOCK_N.
        if H >= 512:
            BLOCK_H = 128
            num_warps = 4
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
        else:
            BLOCK_H = 64 if H >= 64 else 32
            num_warps = 2

        if N >= 2048:
            BLOCK_N = 128
        elif N >= 1024:
            BLOCK_N = 128
        elif N >= 256:
            BLOCK_N = 64
        else:
            BLOCK_N = 32

        num_stages = 2

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        # Launch Triton kernel
        scatter_add_2d_kernel[grid](
            output, expert_outputs, token_indices_32,
            M, H, N,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
