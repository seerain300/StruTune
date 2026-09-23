import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_n_h_kernel(
        output_ptr,          # *bfloat16, shape (M, H)
        expert_ptr,          # *bfloat16, shape (N, H)
        indices_ptr,         # *int32,    shape (N,)
        N: tl.constexpr,     # number of updates
        H: tl.constexpr,     # hidden size
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # 2D launch: axis 0 tiles over N, axis 1 tiles over H.
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Rows in this block
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       # [BLOCK_N]
        mask_n = n_offs < N

        # Hidden feature offsets for this block
        h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)       # [BLOCK_H]
        mask_h = h_offs < H

        # Load token indices for each source row n in this block
        # indices_ptr is int32, convert to int64 for address arithmetic
        idx = tl.load(indices_ptr + n_offs, mask=mask_n, other=0)
        idx = idx.to(tl.int64)  # target row in output

        # Compute 2D pointers into expert_outputs for this block (broadcasted)
        # Shape: (BLOCK_N, BLOCK_H)
        src_ptrs = expert_ptr + n_offs[:, None] * H + h_offs[None, :]  # int64
        mask = mask_n[:, None] & mask_h[None, :]

        # Load the expert values for this block
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        # Destination pointers: output[row=idx, col=h_offs]
        dest_ptrs = output_ptr + idx[:, None] * H + h_offs[None, :]

        # Atomic add to accumulate expert contributions
        tl.atomic_add(dest_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward: perform scatter-add along dim=0.
        output[token_indices[i]] += expert_outputs[i] for all i.

        Args:
            final_hidden_states: (M, H), bfloat16
            expert_outputs:      (N, H), bfloat16
            token_indices:       (N,), int64 or int32

        Returns:
            output: (M, H), bfloat16, with additions
        """
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch if Triton is not available (for robustness).
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Validate shapes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernel."
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be int64 or int32"
        M, H = final_hidden_states.shape
        N, H_exp = expert_outputs.shape
        assert H == H_exp, "hidden_size mismatch between final_hidden_states and expert_outputs"

        # Initialize output (clone) to match index_add behavior
        output = final_hidden_states.clone()

        # Prepare indices as int32 for faster address arithmetic in Triton
        if token_indices.dtype != torch.int32:
            indices_i32 = token_indices.to(torch.int32)
        else:
            indices_i32 = token_indices

        # Choose tiling parameters adaptively
        # Larger BLOCK_H increases per-program parallelism on H, but too large may reduce occupancy.
        H_blocks = (H + 63) // 64  # heuristic for num_warps below
        if H >= 512:
            BLOCK_H = 128
            num_warps = 4
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
        else:
            BLOCK_H = 64
            num_warps = 2

        # Tile over N; small BLOCK_N increases the number of programs and helps distribution
        BLOCK_N = 64 if N >= 64 else 32
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        scatter_add_n_h_kernel[grid](
            output, expert_outputs, indices_i32,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
