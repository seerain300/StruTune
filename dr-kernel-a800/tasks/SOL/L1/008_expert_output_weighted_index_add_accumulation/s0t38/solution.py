import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,       # *bf16 or *fp16, pointer to output [M, H]
    expert_ptr,       # *bf16 or *fp16, pointer to expert_outputs [N, H]
    index_ptr,        # *int32, pointer to token_indices [N]
    M: tl.int32,      # number of rows in output
    H: tl.int32,      # hidden size
    N: tl.int32,      # number of source rows
    BLOCK_H: tl.constexpr,
):
    # One program per source row
    n = tl.program_id(0)
    if n >= N:
        return

    # Process H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load destination row index for this source row
        dest = tl.load(index_ptr + n)  # n < N, so index is in-bounds

        # Compute base offsets
        out_base = dest * H + offs
        exp_base = n * H + offs

        # Load expert values for this chunk (masked for tail)
        vals = tl.load(expert_ptr + exp_base, mask=mask, other=0.0)

        # Atomic add into output
        tl.atomic_add(output_ptr + out_base, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only implementation of scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i]
        Assumes final_hidden_states is the accumulation buffer.
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dim() == 2 and expert_outputs.dim() == 2, "final_hidden_states and expert_outputs must be 2D."
        assert token_indices.dim() == 1, "token_indices must be 1D."

        M, H = final_hidden_states.shape
        N = token_indices.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (N, H)."

        # Triton prefers int32 indices for address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Ensure contiguous layout
        output = final_hidden_states
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Choose chunk size and launch config based on H
        if H >= 4096:
            BLOCK_H = 512
            num_warps = 8
            num_stages = 2
        elif H >= 1024:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 2
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        elif H >= 128:
            BLOCK_H = 64
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Grid: one program per source row
        grid = (N,)

        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
