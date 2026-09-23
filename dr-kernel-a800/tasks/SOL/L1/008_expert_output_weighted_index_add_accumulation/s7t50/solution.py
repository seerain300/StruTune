import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Base pointers for the current destination row and source row
    out_row_ptr = out_ptr + dst * H
    src_row_ptr = src_ptr + pid * H

    # Process hidden dimension in chunks of BLOCK_SIZE with proper masking
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H  # boolean mask for valid elements

        # Load a chunk of src row; masked lanes load 0
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomic add corresponding chunk into out row
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          final_hidden_states[indices[i]] += expert_outputs[i] for all i
        """
        # Ensure inputs are contiguous and on the same CUDA device
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        assert final_hidden_states.dtype == expert_outputs.dtype, "final_hidden_states and expert_outputs must have the same dtype."
        assert final_hidden_states.dim() == 2 and expert_outputs.dim() == 2, "Shapes must be [N, H] and [M, H]."
        N, H = final_hidden_states.shape
        M = expert_outputs.shape[0]
        assert token_indices.shape[0] == M, "token_indices must have length M."

        # Make sure token_indices is int32 for Triton
        token_indices_i32 = token_indices if token_indices.dtype == torch.int32 else token_indices.to(torch.int32)

        # Output buffer: clone to preserve original semantics
        out = final_hidden_states.clone()

        # Launch kernel: one program per source row
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices_i32,
            M, N, H,
            BLOCK_SIZE=128,
            num_warps=2,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
