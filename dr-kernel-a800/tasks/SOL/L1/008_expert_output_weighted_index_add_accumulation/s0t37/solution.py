import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,       # *bf16 or *fp16, pointer to output [M, H]
    expert_ptr,       # *bf16 or *fp16, pointer to expert_outputs [N, H]
    index_ptr,        # *int32, pointer to token_indices [N]
    M, H, N,          # int32 scalars for shape info
    BLOCK_H: tl.constexpr,
):
    # One program per source row n
    n = tl.program_id(0)
    if n >= N:
        return

    # Load destination token index for this source row (int32)
    dest = tl.load(index_ptr + n)

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)         # vector within [0, H)
        mask = offs < H                                # mask for last partial chunk

        # Load current chunk of expert outputs for row n
        # expert_ptr is row-major [N, H] => offset = n * H + offs
        vals = tl.load(expert_ptr + n * H + offs, mask=mask, other=0.0)

        # Atomic add into output at (dest, offs)
        # output_ptr is row-major [M, H] => offset = dest * H + offs
        tl.atomic_add(output_ptr + dest * H + offs, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton-only forward: no PyTorch tensor ops here.
        # Ensure inputs are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be CUDA tensors"

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Cast token_indices to int32 for Triton indexing
        token_indices_int32 = token_indices.to(torch.int32)

        # Choose BLOCK_H adaptively based on H
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

        # Launch one program per source row; accumulate directly into final_hidden_states
        grid = (N,)
        scatter_add_rows_kernel[grid](
            final_hidden_states,  # accumulate in-place
            expert_outputs,       # assume inputs are contiguous as per benchmark setup
            token_indices_int32,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return final_hidden_states


def run(*args):
    return ModelNew()(*args)
