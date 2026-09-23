import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H]
    expert_ptr,            # *bf16 or *fp16, shape [N, H]
    index_ptr,             # *int32, shape [N]
    M, H, N,               # int32 runtime sizes
    BLOCK_H: tl.constexpr,
):
    # One Triton program per source row (n)
    n = tl.program_id(0)

    # Load destination index for this source row
    dest = tl.load(index_ptr + n)  # int32

    # Iterate over H in chunks
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # Compute pointers for this row chunk
        expert_row_ptr = expert_ptr + n * H + h_offsets  # [BLOCK_H]
        out_row_ptr = output_ptr + dest * H + h_offsets  # [BLOCK_H]

        # Load expert outputs and atomic add to output
        vals = tl.load(expert_row_ptr, mask=h_mask, other=0.0)
        tl.atomic_add(out_row_ptr, vals, mask=h_mask)

        start += BLOCK_H


def _choose_row_chunk_params(H: int):
    # Heuristic for BLOCK_H and kernel launch params
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
    return BLOCK_H, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        output = final_hidden_states  # accumulate in-place

        # Ensure tensors are contiguous
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()
        # Use int32 for indices for faster address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()

        # Choose params and launch kernel
        BLOCK_H, num_warps, num_stages = _choose_row_chunk_params(H)
        grid = (N,)
        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages
        )

        return output


def run(*args):
    return ModelNew()(*args)
