import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16, pointer to [M, H] (final_hidden_states to be mutated)
    expert_ptr,            # *bf16, pointer to [N, H] (expert_outputs)
    index_ptr,             # *int32, pointer to [N] (token_indices)
    H,                     # hidden size (runtime int)
    BLOCK_H: tl.constexpr  # chunk size along H
):
    # One Triton program per source row n
    n = tl.program_id(axis=0)

    # Load destination row index for this source row
    dst_row = tl.load(index_ptr + n).to(tl.int32)

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load a chunk of expert outputs for this source row
        expert_row_ptr = expert_ptr + n * H + offs
        vals = tl.load(expert_row_ptr, mask=mask, other=0.0)  # bfloat16

        # Compute destination addresses in output: output[dst_row, offs]
        dst_ptrs = output_ptr + dst_row * H + offs

        # Atomic add into output (mutate in-place)
        tl.atomic_add(dst_ptrs, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        final_hidden_states[token_indices[i]] += expert_outputs[i]
        Shapes:
          final_hidden_states: [M, H], bfloat16 (to be mutated)
          expert_outputs: [N, H], bfloat16
          token_indices: [N], int32 (expects 0 <= token_indices[i] < M)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "Triton kernel requires CUDA tensors"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "This implementation expects bfloat16 tensors"
        assert token_indices.dtype == torch.int32, "token_indices must be int32 for Triton addressing"

        # Ensure contiguity for efficient addressing
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Adaptive kernel configuration based on H
        if H >= 2048:
            BLOCK_H = 256
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 128
            num_warps = 4
        else:
            BLOCK_H = 64
            num_warps = 2

        # Launch kernel: one program per source row
        grid = (N,)
        scatter_add_row_kernel[grid](
            final_hidden_states, expert_outputs, token_indices,
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2
        )
        return final_hidden_states


def run(*args):
    return ModelNew()(*args)
