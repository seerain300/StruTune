import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,           # *const float (final_hidden_states clone), shape (M, H), bfloat16
    expert_ptr,           # *const float, expert_outputs, shape (N, H), bfloat16
    index_ptr,            # *const int32, token_indices, shape (N,), int32
    N: tl.constexpr,      # number of rows in expert_outputs (and number of updates)
    H: tl.constexpr,      # number of hidden features
    BLOCK_H: tl.constexpr # tile size along H
):
    # Each program handles one source row 'n' and iterates over H in BLOCK_H chunks.
    n = tl.program_id(axis=0)
    if n >= N:
        return

    # Load target row index for this source row.
    # index_ptr is int32, token_indices may be int64 in host; we pass int32 to Triton.
    idx = tl.load(index_ptr + n)

    # Iterate over H in chunks and atomic_add each chunk to the target row.
    for h_off in range(0, H, BLOCK_H):
        h_offsets = h_off + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load the current chunk from expert_outputs for row n.
        # expert_outputs is [N, H], row n at columns h_offsets.
        vals = tl.load(expert_ptr + n * H + h_offsets, mask=mask, other=0.0)

        # Compute destination addresses in output: row idx, columns h_offsets.
        dest = idx * H + h_offsets
        # Atomic add the chunk into output.
        tl.atomic_add(output_ptr + dest, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone the input buffer to serve as the output accumulator. This matches original semantics.
        output = final_hidden_states.clone()

        # Ensure token_indices is on the same device and int32 for Triton. (PyTorch indices are int64 by default.)
        # We keep output dtype as bfloat16 to match original.
        indices_i32 = token_indices.to(torch.int32)

        # Number of updates and hidden size
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Choose BLOCK_H and launch parameters based on H for better performance.
        if H >= 256:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 2
        elif H >= 128:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Grid: one program per source row
        grid = (N,)

        # Launch Triton kernel
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices_i32,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
