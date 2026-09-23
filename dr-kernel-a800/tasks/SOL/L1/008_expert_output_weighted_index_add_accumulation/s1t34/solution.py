import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    N: tl.constexpr,  # number of source rows
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size across H
):
    # One program per source row
    row = tl.program_id(0)
    if row >= N:
        return

    # Load token index (int32)
    idx = tl.load(indices_ptr + row)

    # Vector of hidden offsets for this tile
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    # Compute pointers for this row
    out_row_ptr = out_ptr + idx * H + offs
    src_row_ptr = src_ptr + row * H + offs

    # Load source values and atomically accumulate into output
    val = tl.load(src_row_ptr, mask=mask, other=0.0)
    # out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)  # not needed for atomic_add with addend
    tl.atomic_add(out_row_ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA for Triton"
        # Ensure dtypes and contiguity
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors"
        assert token_indices.dtype == torch.int32, "token_indices must be int32"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone final_hidden_states to output
        output = final_hidden_states.clone()

        M = output.shape[0]
        N = expert_outputs.shape[0]
        H = output.shape[1]

        # Choose a robust tile size: 256 balances occupancy and vector width
        BLOCK_H = 256
        grid = (N,)

        scatter_add_rows_kernel[grid](
            output,               # out_ptr
            expert_outputs,       # src_ptr
            token_indices,        # indices_ptr
            N=N,                  # constexpr
            H=H,                  # constexpr
            BLOCK_H=BLOCK_H,      # constexpr
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
