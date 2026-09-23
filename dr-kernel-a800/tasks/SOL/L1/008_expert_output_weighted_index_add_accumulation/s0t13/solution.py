import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_H': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_H': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_H': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_H': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'H']
)
@triton.jit
def scatter_add_exact_2d_kernel(
    output_ptr,            # *output* (M, H), dtype: bfloat16
    expert_outputs_ptr,    # *expert_outputs* (N, H), dtype: bfloat16
    token_indices_ptr,     # *token_indices* (N,), dtype: int64
    N: tl.constexpr,       # number of expert outputs
    H: tl.constexpr,       # hidden size
    BLOCK_N: tl.constexpr, # tile size along N
    BLOCK_H: tl.constexpr, # tile size along H
):
    # Each program processes a chunk of N rows and a chunk of H hidden features.
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Offsets within this tile
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)   # (BLOCK_H,)

    # Boundary masks
    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load token indices for these rows
    # token_indices are int64
    idx = tl.load(token_indices_ptr + n_offsets, mask=mask_n, other=0)  # int64
    idx = idx.to(tl.int64)  # ensure int64 for pointer arithmetic

    # For each row in the tile, add the corresponding expert_output row to the destination
    # We avoid atomics and perform deterministic accumulation.
    for r in range(BLOCK_N):
        n_idx = n_offsets[r]
        row_valid = n_idx < N  # scalar boolean

        # Load expert_outputs[n_idx, h_offsets]
        vals = tl.load(expert_outputs_ptr + n_idx * H + h_offsets,
                       mask=mask_h & row_valid,
                       other=0)  # (BLOCK_H,) bfloat16

        # Destination row pointer: output[idx[r], h_offsets]
        dest_row_ptr = output_ptr + (idx[r] * H + h_offsets)

        # Load current output and add vals, then store back
        curr = tl.load(dest_row_ptr, mask=mask_h & row_valid, other=0)
        tl.store(dest_row_ptr, curr + vals, mask=mask_h & row_valid)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Performs scatter-add along dim=0: output[token_indices[i]] += expert_outputs[i]
        using a Triton kernel for performance while keeping exact numerical behavior.
        """
        # Ensure all tensors are on the same device
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be CUDA tensors"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Inputs must be bfloat16 as in the original code"

        # Shape checks
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        # token_indices length must be N
        assert token_indices.numel() == N, "token_indices length must match num_selected_tokens"

        # Clone to ensure we don't modify input in-place
        output = final_hidden_states.clone()

        # Ensure token_indices dtype is int64 (PyTorch index_add uses long)
        if token_indices.dtype != torch.long:
            token_indices = token_indices.to(torch.long)

        # Launch Triton kernel with a grid that depends on the selected config
        # Grid: (ceil_div(N, BLOCK_N), ceil_div(H, BLOCK_H))
        scatter_add_exact_2d_kernel[
            lambda meta: (triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(H, meta['BLOCK_H']))
        ](
            output, expert_outputs, token_indices,
            N, H,
        )

        return output


def run(*args):
    return ModelNew()(*args)
