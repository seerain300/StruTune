import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,        # *const bfloat16, shape [M, H]
    src_ptr,        # *const bfloat16, shape [N, H]
    indices_ptr,    # *const int32,    shape [N]
    M,              # int: number of rows in output
    H,              # int: hidden size (columns)
    N,              # int: number of source rows
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row
    row_id = tl.program_id(0)  # in [0, N)
    # Load token index (int32)
    idx = tl.load(indices_ptr + row_id)
    # Compute base pointers for this row
    out_row_ptr = out_ptr + idx * H
    src_row_ptr = src_ptr + row_id * H

    # Iterate over hidden dimension in tiles of size BLOCK_H
    # This loop ensures correctness for any H
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load the source vector (bfloat16)
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomically add into the output at row 'idx'
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure dtypes and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 for outputs and sources."
        out = final_hidden_states.clone()  # keep clone to match semantics

        # Make inputs contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        M = out.shape[0]
        H = out.shape[1]
        N = expert_outputs.shape[0]

        # Select tile size and warps based on H for performance
        if H <= 512:
            BLOCK_H = 512
            num_warps = 8
        elif H <= 1024:
            BLOCK_H = 1024
            num_warps = 8
        else:
            BLOCK_H = 256
            num_warps = 4

        # Launch grid: one program per source row
        grid = (N,)

        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices, M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
