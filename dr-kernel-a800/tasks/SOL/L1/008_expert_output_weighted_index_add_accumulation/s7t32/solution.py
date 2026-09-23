import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index for this source row
    dst = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute pointers for this chunk (row-major contiguous [N, H])
        out_row_ptr = out_ptr + dst * H + offs
        src_row_ptr = src_ptr + pid * H + offs

        # Atomic add for each element in the chunk
        # Mask handles partial tails when H is not a multiple of BLOCK_SIZE.
        tl.atomic_add(out_row_ptr, tl.load(src_row_ptr, mask=mask, other=0.0), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device."
        out = final_hidden_states.clone()
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Shapes
        N, H = out.shape  # out shape is [N, H]
        M = expert_outputs.shape[0]  # number of expert outputs to scatter

        # Launch Triton kernel: one program per source row
        grid = (M,)
        # Use a moderate BLOCK_SIZE; 128 works well for many H, and masking handles partial tails.
        BLOCK_SIZE = 128
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H, BLOCK_SIZE,
            num_warps=1, num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
