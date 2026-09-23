import torch
import triton
import triton.language as tl


# Triton kernel: for each token i, add expert_outputs[i] to output[token_indices[i]]
# Assumes:
# - output is [M, N], cloned from final_hidden_states
# - expert_outputs is [K, N]
# - token_indices is [K] as int32
@triton.jit
def index_add_dim0_kernel(output_ptr, expert_ptr, indices_ptr,
                          M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                          BLOCK_H: tl.constexpr):
    # One program per token
    pid = tl.program_id(axis=0)
    # bounds check: if pid >= K, do nothing
    if pid >= K:
        return

    # Load index for this token
    idx = tl.load(indices_ptr + pid)  # int32
    # Compute base offsets for rows
    row_out = idx * N
    row_exp = pid * N

    # Vector of column offsets
    cols = tl.arange(0, BLOCK_H)

    # Loop over columns in chunks of BLOCK_H
    # Note: We use a simple while loop to support any N
    start = 0
    while start < N:
        col_offsets = start + cols
        mask = col_offsets < N
        # Load current output row slice
        out_vals = tl.load(output_ptr + row_out + col_offsets, mask=mask, other=0.0)
        # Load expert row slice
        src_vals = tl.load(expert_ptr + row_exp + col_offsets, mask=mask, other=0.0)
        # Add
        out_vals = out_vals + src_vals
        # Store back
        tl.store(output_ptr + row_out + col_offsets, out_vals, mask=mask)
        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Replicates:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, token_indices, expert_outputs)
        Using Triton for the index_add, PyTorch for the clone to guarantee exactness.
        """
        # Ensure dtypes and devices are consistent
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Dtype must be bfloat16."

        # Clone to match reference semantics exactly
        output = final_hidden_states.clone()

        # Cast indices to int32 for Triton
        # Note: token_indices are typically long (int64); Triton prefers int32 for indexing.
        indices_i32 = token_indices.to(torch.int32)

        # Shapes
        M = final_hidden_states.shape[0]
        K = expert_outputs.shape[0]
        N = final_hidden_states.shape[1]

        # Launch Triton kernel: one program per token
        grid = (K,)
        # Choose a reasonable BLOCK_H for vectorization along N.
        # Using 128 or 256 works well across common hidden sizes. We pick 128 to reduce register pressure.
        BLOCK_H = 128
        index_add_dim0_kernel[grid](
            output, expert_outputs, indices_i32,
            M=M, K=K, N=N,
            BLOCK_H=BLOCK_H,
            num_warps=4,  # typical choice; can tune based on N
            num_stages=2  # typical choice; can tune based on N
        )

        return output


def run(*args):
    return ModelNew()(*args)
