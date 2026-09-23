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

    # Destination row index for this source row
    dest = tl.load(index_ptr + n)  # int32
    # Base pointers for this source row
    expert_row_ptr = expert_ptr + n * H
    # Iterate over H in chunks
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)  # [BLOCK_H] int32
        mask = offs < H
        # Load expert values for this chunk (coalesced along H)
        vals = tl.load(expert_row_ptr + offs, mask=mask, other=0.0)
        # Atomic add into output at destination row
        out_row_ptr = output_ptr + dest * H
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)
        start += BLOCK_H


def _choose_launch_params(H: int):
    # Adaptive tuning based on H
    if H >= 2048:
        return 256, 4, 3  # BLOCK_H, num_warps, num_stages
    elif H >= 512:
        return 256, 4, 2
    elif H >= 128:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton-only implementation: no PyTorch tensor ops in host code
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA"
        # Ensure dtype compatibility; we will operate in the same dtype as inputs (bf16 in provided setup)
        output = final_hidden_states
        # Token indices as int32 for Triton address arithmetic
        if token_indices.dtype != torch.int32:
            index_i32 = token_indices.to(torch.int32)
        else:
            index_i32 = token_indices

        M, H = output.shape
        N = expert_outputs.shape[0]
        # Choose kernel params based on H
        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)
        grid = (N,)  # one program per source row

        scatter_add_rows_kernel[grid](
            output, expert_outputs, index_i32,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
