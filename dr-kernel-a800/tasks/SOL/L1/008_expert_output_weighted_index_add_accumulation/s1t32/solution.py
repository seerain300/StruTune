import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    N: tl.constexpr,  # number of rows in src_ptr / number of additions
    H: tl.constexpr,  # hidden size (columns)
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load the target index for this row
    idx = tl.load(indices_ptr + i)  # int32

    # Iterate across hidden dimension in tiles
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)  # vector of column offsets
        mask = offs < H

        # Load source values (bfloat16) for this row and tile
        vals = tl.load(src_ptr + i * H + offs, mask=mask, other=0.0)

        # Atomic add into output at the selected row
        tl.atomic_add(out_ptr + idx * H + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 for outputs and source."
        assert token_indices.dtype == torch.int64, "token_indices expected to be torch.long (int64)."

        # Clone for output
        output = final_hidden_states.clone()
        # Make sure tensors are contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Grid: one program per source row
        grid = (N,)

        # Fixed tile size that balances performance and register usage
        BLOCK_H = 256
        num_warps = 4
        num_stages = 2

        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            N=N, H=H, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages
        )
        return output


def run(*args):
    return ModelNew()(*args)
