import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,      # *bf16/float16 pointer to output [M, H]
    expert_ptr,      # *bf16/float16 pointer to expert_outputs [N, H]
    index_ptr,       # *int32 pointer to token_indices [N]
    M: tl.constexpr, # number of rows in output
    N: tl.constexpr, # number of source rows
    H: tl.constexpr, # number of hidden features
    BLOCK_H: tl.constexpr,
):
    # One program handles one source row n in [0, N)
    n = tl.program_id(0)
    if n >= N:
        return

    # Process H in chunks of BLOCK_H
    h = 0
    while h < H:
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load token index for this source row n
        idx = tl.load(index_ptr + n)  # int32

        # Load expert outputs for this row n and chunk h_offsets
        # expert layout is row-major: address = n * H + h_offsets
        expert_row_ptr = expert_ptr + n * H
        vals = tl.load(expert_row_ptr + h_offsets, mask=mask, other=0.0)

        # Atomic add the chunk into output at row idx
        out_row_ptr = output_ptr + idx * H
        tl.atomic_add(out_row_ptr + h_offsets, vals, mask=mask)

        h += BLOCK_H


def _choose_block_and_warps(H: int):
    # Heuristic: larger H -> larger BLOCK_H for better throughput, but avoid excessive register pressure
    if H >= 256:
        return 256, 8, 2
    elif H >= 128:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          final_hidden_states: [M, H] (bfloat16)
          expert_outputs: [N, H] (bfloat16)
          token_indices: [N] (int64/int32), positions in [0, M)
        Returns:
          output: [M, H] updated with atomic adds
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 tensors."
        # Ensure contiguity for coalesced access
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs's hidden size must match output's hidden size."

        # Triton expects int32 for index math; cast if needed
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        assert token_indices.shape[0] == N and token_indices.min().item() >= 0 and token_indices.max().item() < M, "token_indices out of range."

        # Launch kernel: one program per source row
        grid = (N,)
        BLOCK_H, num_warps, num_stages = _choose_block_and_warps(H)

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M=M, N=N, H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
