import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,     shape [N]
    N,                # total source rows
    H,                # hidden size
    BLOCK_H: tl.constexpr,  # tile size over H
):
    # One program per source row
    row = tl.program_id(0)  # 0 .. N-1
    # Load destination row index and source vector for this row
    # Destination row index may be out of range if N > M, but here N = M * num_experts_per_tok, so it is valid.
    dest_row = tl.load(indices_ptr + row)  # int32
    # We will iterate over H in tiles
    for j in range(0, H, BLOCK_H):
        offs = j + tl.arange(0, BLOCK_H)
        mask = offs < H
        # Compute pointers
        src_row_ptr = src_ptr + row * H + offs
        out_row_ptr = out_ptr + dest_row * H + offs
        # Load source values (bf16)
        val = tl.load(src_row_ptr, mask=mask, other=0.0)
        # Atomic add into output
        tl.atomic_add(out_row_ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add equivalent to:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        Returns the updated output.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors"
        # Ensure contiguous tensors and dtypes
        out = final_hidden_states.clone()  # this performs the initial clone
        # Make inputs contiguous and cast indices to int32
        src = expert_outputs.contiguous()
        idx = token_indices.contiguous()
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)

        M = out.shape[0]
        H = out.shape[1]
        N = src.shape[0]
        # Launch kernel: one program per source row
        grid = (N,)
        # Use BLOCK_H=256, num_warps=4, num_stages=2 for good balance on common H<=1024
        scatter_add_rows_kernel[grid](
            out, src, idx,
            N, H,
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
