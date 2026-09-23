import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # number of rows in output
    N: tl.constexpr,  # number of source rows (not used for bounds, but can be for verification)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size across H
):
    # One program per source row i
    i = tl.program_id(0)
    # Load the target row index for this source row
    idx = tl.load(indices_ptr + i)  # int32

    # Iterate across hidden dimension in tiles of size BLOCK_H
    for offs in range(0, H, BLOCK_H):
        col = offs + tl.arange(0, BLOCK_H)  # vector of column indices
        mask = col < H

        # Load expert_outputs[i, col]
        src_row_ptr = src_ptr + i * H + col
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Compute destination pointers for output[idx, col]
        out_row_ptr = out_ptr + idx * H + col
        # Atomic add into output
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of scatter-add:
        output = final_hidden_states.clone()
        for i in range(N): output[token_indices[i]] += expert_outputs[i]
        """
        assert final_hidden_states.is_cuda, "Tensors must be on CUDA device for Triton kernel"
        assert expert_outputs.is_cuda, "Tensors must be on CUDA device for Triton kernel"
        assert token_indices.is_cuda, "Tensors must be on CUDA device for Triton kernel"

        # Ensure dtypes and contiguity
        out = final_hidden_states.clone()  # output will be mutated
        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = out.shape[0]
        N, H = expert_outputs.shape
        assert token_indices.shape[0] == N, "token_indices must have length N = M * num_experts_per_tok"

        # Kernel launch configuration
        BLOCK_H = 256
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out,
            expert_outputs,
            token_indices,
            M=M,
            N=N,
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
