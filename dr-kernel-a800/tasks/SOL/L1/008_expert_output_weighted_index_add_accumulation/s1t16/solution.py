import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,     shape [N]
    N,                # int32: number of rows in src (length of indices)
    H,                # int32: hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row i
    row = tl.program_id(0)
    if row >= N:
        return

    # Load the token index for this row
    idx = tl.load(indices_ptr + row)  # int32

    # Iterate over H in tiles and perform vectorized atomic_add
    j = 0
    while j < H:
        offsets = j + tl.arange(0, BLOCK_H)
        mask = offsets < H

        # Compute linear offsets for source and output rows
        src_offsets = row * H + offsets
        out_offsets = idx * H + offsets

        # Load source values (masked for tail)
        vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

        # Atomic add into output; idx may repeat, so atomic is required
        tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)

        j += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure all tensors are on CUDA and Triton-friendly
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        out = final_hidden_states.clone()
        # Ensure dtypes consistent (bfloat16)
        if out.dtype != torch.bfloat16:
            out = out.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)
        # Indices as int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Ensure contiguity
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        M = out.shape[0]
        H = out.shape[1]
        N = expert_outputs.shape[0]

        # Launch one program per source row
        grid = (N,)
        # Tuned parameters from evaluation (best-performing configuration)
        BLOCK_H = 256
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
