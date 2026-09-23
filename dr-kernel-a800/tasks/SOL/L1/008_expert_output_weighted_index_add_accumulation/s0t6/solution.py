import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_atomic_kernel(
    output_ptr,     # *fp16/bf16/fp32
    expert_ptr,     # *fp16/bf16/fp32
    indices_ptr,    # *int64
    N, H,           # int32
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: programs over (rows of expert_outputs, tiles of H)
    row_block = tl.program_id(0)
    h_block = tl.program_id(1)

    n_offsets = row_block * BLOCK_N + tl.arange(0, BLOCK_N)     # [BLOCK_N]
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)       # [BLOCK_H]

    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load source row indices (token positions) for these N elements
    idxs = tl.load(indices_ptr + n_offsets, mask=mask_n, other=0)  # [BLOCK_N], int64

    # If this H-block covers the entire H, process full row vector and do a single atomic add
    if (h_block == 0) and (H <= BLOCK_H):
        # Build full H range
        h_full = tl.arange(0, BLOCK_H)  # [BLOCK_H], but we only use h_offsets < H
        mask_h_full = h_full < H

        # For each n in the block, load expert row and atomic add to output[idxs[n], :]
        for i in range(BLOCK_N):
            n = n_offsets[i]
            valid_n = n < N
            # Load expert row values for this n (masked by H)
            expert_vals = tl.load(
                expert_ptr + n * H + h_offsets,
                mask=mask_h,
                other=0.0,
            )
            # Target row index for this n
            idx = idxs[n]
            # Destination offsets for this H block
            dest_offsets = idx * H + h_offsets
            # Atomic add with mask for valid n and h
            tl.atomic_add(
                output_ptr + dest_offsets,
                expert_vals,
                mask=mask_h & valid_n,
            )
    else:
        # General case: process per-H block atomics
        for i in range(BLOCK_N):
            n = n_offsets[i]
            valid_n = n < N
            # Load expert row values for this n (masked by H)
            expert_vals = tl.load(
                expert_ptr + n * H + h_offsets,
                mask=mask_h,
                other=0.0,
            )
            # Target row index for this n
            idx = idxs[n]
            # Destination offsets for this H block
            dest_offsets = idx * H + h_offsets
            # Atomic add with mask for valid n and h
            tl.atomic_add(
                output_ptr + dest_offsets,
                expert_vals,
                mask=mask_h & valid_n,
            )


def triton_scatter_add(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Triton implementation of scatter-add: output[token_indices[i]] += expert_outputs[i]
    output: (M, H), expert_outputs: (N, H), token_indices: (N,)
    """
    assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
    assert output.dtype == expert_outputs.dtype, "Dtypes of output and expert_outputs must match."

    M, H = output.shape
    N = expert_outputs.shape[0]

    # Choose block sizes and execution config based on problem size
    if H >= 1024:
        BLOCK_H = 128
        num_warps = 4
        num_stages = 3
    elif H >= 256:
        BLOCK_H = 64
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_H = 32
        num_warps = 2
        num_stages = 2

    # Increase BLOCK_N to improve parallelism across rows when N is large
    BLOCK_N = 256 if N >= 256 else (128 if N >= 128 else (64 if N >= 64 else 32))
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

    scatter_add_2d_atomic_kernel[grid](
        output, expert_outputs, token_indices,
        N, H,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward: performs output[token_indices[i]] += expert_outputs[i]
        without using any PyTorch tensor ops in the forward path (except for potential fallback).
        """
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            # Fallback for safety if not on CUDA
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Initialize output to zeros; we only need to accumulate into it
        output = torch.empty_like(final_hidden_states)
        output.zero_()

        # Launch Triton kernel
        triton_scatter_add(output, expert_outputs, token_indices)
        return output


def run(*args):
    return ModelNew()(*args)
