import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token (i). Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_per_index_kernel(
    output_ptr,         # *ptr to output [M, H]
    expert_ptr,         # *ptr to expert_outputs [N_selected, H]
    indices_ptr,        # *ptr to token_indices [N_selected] (int32)
    M,                  # number of rows in output (i.e., batch_size * seq_len)
    H,                  # hidden_size (number of columns)
    N,                  # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size along hidden dimension
):
    # Program id corresponds to the selected token index
    i = tl.program_id(axis=0)
    # Load the target row index for this selected token
    row_index = tl.load(indices_ptr + i)

    # Base pointers for this selected token's row in expert and for the output row
    expert_row_base = expert_ptr + i * H
    output_row_base = output_ptr + row_index * H

    # Iterate over the hidden dimension in chunks of BLOCK_SIZE
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load expert row chunk (bfloat16), masked for tail
        vals = tl.load(expert_row_base + offs, mask=mask, other=0.0)

        # Atomic add into output row chunk
        tl.atomic_add(output_row_base + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add:
        - Clone final_hidden_states to output.
        - For each selected token i, add expert_outputs[i, :] to output[token_indices[i], :].
        """
        # Ensure contiguity and dtype compatibility; we keep output dtype as bfloat16 to match inputs
        output = final_hidden_states.clone()
        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per selected token
        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        if TRITON_AVAILABLE and output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            BLOCK_SIZE = 256
            grid = (N,)
            scatter_add_per_index_kernel[grid](
                output, expert_outputs, token_indices,
                M, H, N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=1,
            )
        else:
            # Fallback: PyTorch scatter-add (kept for robustness if Triton/CUDA unavailable)
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
