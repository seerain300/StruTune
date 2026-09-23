import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H]
    src_ptr,        # *bf16, pointer to src tensor [M, H]
    indices_ptr,    # *int32, pointer to indices tensor [M]
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size for hidden dim
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)

    # Base pointers for this row
    out_row_ptr = out_ptr + dst * H
    src_row_ptr = src_ptr + pid * H

    # Iterate over hidden dimension in chunks
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load source values for this chunk (masked)
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)
        # Atomic add into out at destination row
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match original behavior
        out = final_hidden_states.clone()

        # Ensure contiguity and device
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Shapes
        N = out.shape[0]
        M = expert_outputs.shape[0]
        H = out.shape[1]

        # Launch grid: one program per source row
        grid = (M,)

        # Larger chunk size to reduce loop iterations; masked loop handles partial tails
        BLOCK_SIZE = 256

        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
