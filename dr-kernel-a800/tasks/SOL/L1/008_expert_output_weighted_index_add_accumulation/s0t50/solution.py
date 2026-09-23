import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16 pointer, shape [M, H]
    expert_ptr,            # *bf16 or *fp16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    M: tl.constexpr,       # number of rows in output
    H: tl.constexpr,       # hidden size
    BLOCK_H: tl.constexpr  # chunk size over H
):
    # One program per source row (n in [0, N))
    n = tl.program_id(0)

    # Load destination row index for this source row
    dest_row = tl.load(index_ptr + n)
    # Defensive: skip if out of range
    if dest_row < 0 or dest_row >= M:
        return

    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load expert values for this row chunk
        vals = tl.load(expert_ptr + n * H + h_offsets, mask=mask, other=0.0)

        # Atomic add to the corresponding output row chunk
        out_ptrs = output_ptr + dest_row * H + h_offsets
        tl.atomic_add(out_ptrs, vals, mask=mask)

        start += BLOCK_H


def _choose_block_params(H: int):
    # Heuristic tuning for block size and launch config
    # Larger H -> larger chunk and more warps for better throughput
    if H >= 1024:
        return 256, 8, 3
    elif H >= 256:
        return 128, 4, 3
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        output[token_indices[i]] += expert_outputs[i]
        - output: (M, H), dtype bfloat16/fp16
        - expert_outputs: (N, H)
        - token_indices: (N,) int64; converted to int32 for kernel
        """
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors."
        output = final_hidden_states
        # Ensure contiguous row-major layout
        if not output.is_contiguous():
            output = output.contiguous()
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()

        # Convert indices to int32 for efficient address arithmetic
        index32 = token_indices.to(torch.int32)

        M = output.shape[0]
        H = output.shape[1]
        N = index32.shape[0]

        BLOCK_H, num_warps, num_stages = _choose_block_params(H)

        # Launch one program per source row
        grid = (N,)
        scatter_add_row_kernel[grid](
            output, expert_outputs, index32,
            M, H, BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
