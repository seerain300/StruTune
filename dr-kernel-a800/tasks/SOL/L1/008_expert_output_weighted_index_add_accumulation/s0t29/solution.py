import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16 pointer, shape [M, H]
    expert_ptr,            # *bf16 or *fp16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    N, H,                  # runtime sizes
    BLOCK_H: tl.constexpr  # compile-time tile size for H
):
    # One program per source row n
    n = tl.program_id(axis=0)
    if n >= N:
        return

    # Load target row index for this source row
    idx = tl.load(index_ptr + n)

    # Iterate over hidden features in chunks of BLOCK_H
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load the corresponding hidden values from the expert output row n
        vals = tl.load(
            expert_ptr + n * H + h_offsets,
            mask=mask,
            other=0.0  # zero for out-of-range elements
        )

        # Compute output destinations: each element goes to output[idx, h_offsets]
        dest = idx * H + h_offsets

        # Atomic add to output
        tl.atomic_add(output_ptr + dest, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only forward that performs scatter-add: output[token_indices[i]] += expert_outputs[i]
        """
        # Clone input buffer to initialize output; we'll modify this clone.
        output = final_hidden_states.clone()

        # Ensure device and dtype consistency
        if not output.is_cuda or not expert_outputs.is_cuda or not token_indices.is_cuda:
            # If tensors are not on CUDA, fall back to torch implementation (for safety)
            # Note: This forward method is expected to run on CUDA devices for Triton performance.
            raise RuntimeError("ModelNew requires CUDA tensors for Triton execution.")

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]

        # Ensure token_indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices_i32 = token_indices.to(torch.int32)
        else:
            token_indices_i32 = token_indices

        # Choose tile size and warps based on H
        if H >= 1024:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Launch Triton kernel: one program per source row
        grid = (N,)

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices_i32,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
