import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per index. Vectorize across hidden_size in chunks of BLOCK_SIZE.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,           # pointer to output tensor (bfloat16), shape [M, H]
    expert_ptr,           # pointer to expert_outputs tensor (bfloat16), shape [N, H]
    indices_ptr,          # pointer to token_indices tensor (int32), shape [N]
    M,                    # int: number of rows in output
    N,                    # int: number of expert outputs
    H,                    # int: hidden_size (number of columns)
    BLOCK_SIZE: tl.constexpr,  # chunk size for columns, power-of-two up to 1024
):
    pid = tl.program_id(axis=0)  # program id over indices
    if pid >= N:
        return

    # Load destination row index
    idx = tl.load(indices_ptr + pid).to(tl.int32)

    # Iterate over hidden_size in chunks
    j = 0
    while j < H:
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Compute pointers for the current chunk in row-major [*, H] contiguous layout
        out_ptrs = output_ptr + idx * H + cols
        exp_ptrs = expert_ptr + pid * H + cols

        # Load chunk and atomic add to output
        vals = tl.load(exp_ptrs, mask=mask, other=0)
        tl.atomic_add(out_ptrs, vals, mask=mask)

        j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run:
        - Clones final_hidden_states (to match original behavior)
        - Accumulates expert_outputs into output[token_indices[i]] via atomic add in Triton
        """
        # Fallback to PyTorch if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Validate dtypes and devices; original uses bfloat16
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype == torch.long, "token_indices must be torch.long"

        # Clone to preserve original semantics
        output = final_hidden_states.clone()

        # Prepare indices as int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Shapes
        M = output.shape[0]  # batch_size * seq_len
        N = expert_outputs.shape[0]
        H = output.shape[1]  # hidden_size

        # Choose a power-of-two BLOCK_SIZE up to 1024 to reduce loop iterations and improve vectorization
        if H <= 1:
            block_size = 1
        else:
            block_size = 1 << ((H - 1).bit_length())  # next power of two
            block_size = min(block_size, 1024)

        # Launch grid: one program per expert output
        grid = (N,)

        # Run Triton kernel
        scatter_add_rows_kernel[grid](
            output,                    # output_ptr
            expert_outputs,            # expert_ptr
            indices_i32,               # indices_ptr
            M, N, H,                   # dimensions
            BLOCK_SIZE=block_size,     # constexpr meta-parameter
            num_warps=4,               # balanced default for memory-bound kernels
            num_stages=2,              # modest pipelining
        )

        return output


def run(*args):
    return ModelNew()(*args)
