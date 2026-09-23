import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16, shape [M, H], contiguous
    expert_ptr,            # *bf16, shape [N, H], contiguous
    index_ptr,             # *int32, shape [N]
    M: tl.constexpr,       # number of rows in output (batch_size * seq_len)
    H: tl.constexpr,       # number of hidden features
    BLOCK_H: tl.constexpr, # chunk size along H
):
    # Each program handles one source row i (0 <= i < N)
    i = tl.program_id(axis=0)

    # If i >= N (defensive, though grid ensures i < N)
    if i >= tl.num_programs(axis=0):
        return

    # Load the destination row index for this source row
    token = tl.load(index_ptr + i)  # int32
    # Convert to 64-bit for pointer arithmetic safety
    token64 = token.to(tl.int64)
    M64 = M.to(tl.int64)
    H64 = H.to(tl.int64)

    # Iterate over H in chunks
    h_start = 0
    while h_start < H:
        h_offsets = h_start + tl.arange(0, BLOCK_H)  # vector of positions along H
        mask = h_offsets < H  # mask for the last chunk

        # Compute output row base pointer: row_offset = token * H
        row_offset = token64 * H64

        # Compute output addresses for this chunk
        out_ptrs = output_ptr + row_offset + h_offsets  # broadcasting adds to vector
        # Load expert outputs for this source row and chunk
        expert_row_ptr = expert_ptr + i * H  # since expert is [N, H], row-major
        expert_vals = tl.load(expert_row_ptr + h_offsets, mask=mask, other=0.0)  # bf16

        # Atomic add into output
        # Note: Triton will cast types appropriately; here output_ptr is *bf16 and expert_vals are *bf16
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)

        h_start += BLOCK_H


def _launch_scatter_add(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Launch the Triton kernel to perform: output[token_indices[i]] += expert_outputs[i]
    output: (M, H), bf16, contiguous
    expert_outputs: (N, H), bf16, contiguous
    token_indices: (N,), int32 (preferred), but int64 also fine; we'll cast to int32 for Triton.
    """
    assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert output.is_contiguous() and expert_outputs.is_contiguous()
    M, H = output.shape
    N = expert_outputs.shape[0]
    # Triton expects int32 indices for simpler arithmetic; cast if needed
    if token_indices.dtype != torch.int32:
        token_indices = token_indices.to(torch.int32)

    # Choose BLOCK_H based on H
    if H >= 2048:
        BLOCK_H = 256
        num_warps = 8
        num_stages = 3
    elif H >= 512:
        BLOCK_H = 128
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_H = 64
        num_warps = 2
        num_stages = 2

    # Launch one program per source row
    grid = (N,)

    scatter_add_row_kernel[grid](
        output,                 # *bf16
        expert_outputs,         # *bf16
        token_indices,          # *int32
        M=M,
        H=H,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add:
          final_hidden_states[i, :] += expert_outputs[token_indices[i], :]
        Entry point required by the evaluation harness.
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Launch the Triton kernel
        return _launch_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
