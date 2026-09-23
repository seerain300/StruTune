import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,          # *bf16, [M, H]
    expert_ptr,          # *bf16, [N, H]
    index_ptr,           # *int32, [N]
    M,                   # int32, number of rows in output
    H,                   # int32, number of columns in output
    N,                   # int32, number of source rows (len of token_indices)
    BLOCK_H: tl.constexpr,
):
    n = tl.program_id(axis=0)  # one program per source row
    if n >= N:
        return

    # Process the H dimension in chunks of BLOCK_H
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)  # int32 vector
        mask = h_offsets < H

        # Load token index for this source row (int32)
        token = tl.load(index_ptr + n)  # int32

        # Load expert output chunk [BLOCK_H]
        expert_vals = tl.load(expert_ptr + n * H + h_offsets, mask=mask, other=0.0)

        # Compute destination addresses: output[token, h_offsets]
        # Use 64-bit for pointer arithmetic safety
        row_base = (token.to(tl.int64)) * H
        dest = row_base + h_offsets.to(tl.int64)

        # Atomic add to output
        tl.atomic_add(output_ptr + dest, expert_vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be CUDA tensors."
        assert output.dtype == torch.bfloat16, "This implementation expects output in bfloat16."
        assert expert_outputs.dtype == torch.bfloat16, "Expert outputs must be bfloat16."
        assert token_indices.dtype in (torch.int32, torch.int64), "Token indices must be integer type."

        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Triton expects int32 indices for efficient arithmetic; cast if needed
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch one program per source row
        grid = (N,)
        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H, N,
            BLOCK_H=128,
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
