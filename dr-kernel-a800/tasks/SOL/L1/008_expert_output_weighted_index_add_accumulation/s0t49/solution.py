import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, pointer to output (M, H)
    expert_ptr,            # *bf16 or *fp16, pointer to expert outputs (N, H)
    index_ptr,             # *int32, pointer to token indices (N,)
    M, N, H,               # int32 runtime sizes
    BLOCK_H: tl.constexpr  # tile size along H
):
    # One program per source row (n)
    n = tl.program_id(0)
    if n >= N:
        return

    # Load token index for this source row (int32)
    idx = tl.load(index_ptr + n)

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load expert_outputs[n, offs] as a vector
        # expert is laid out row-major: row n spans indices [n*H, n*H + H)
        vals = tl.load(expert_ptr + n * H + offs, mask=mask, other=0)

        # Compute output addresses: output[idx, offs]
        out_offs = idx * H + offs
        # Atomic add into output
        tl.atomic_add(output_ptr + out_offs, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of output.index_add_(0, token_indices, expert_outputs)
        that performs atomic accumulation of expert outputs back to token positions.
        """
        # Ensure tensors are on CUDA and contiguous
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            # Fallback: run on CPU if not on CUDA (evaluation uses CUDA, but keep safe)
            return final_hidden_states.index_add_(0, token_indices, expert_outputs)

        # Ensure dtype consistency (bf16 in provided setup)
        if final_hidden_states.dtype != expert_outputs.dtype:
            expert_outputs = expert_outputs.to(final_hidden_states.dtype)

        # Ensure both tensors are contiguous row-major
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Ensure token_indices is int32 for Triton address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M, H = output.shape
        N = token_indices.numel()

        # Choose BLOCK_H and launch config based on H
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
            output, expert_outputs, token_indices, M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
