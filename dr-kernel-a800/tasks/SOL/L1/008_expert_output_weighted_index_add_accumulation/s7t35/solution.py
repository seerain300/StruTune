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
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Pointer to the current slice in the source row
        src_row_ptr = src_ptr + pid * H
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # *bf16
        # Pointer to the destination row in the output
        out_row_ptr = out_ptr + dst * H
        # Atomic add each element; Triton will broadcast pointer offsets
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity and dtypes
        out = final_hidden_states.clone().contiguous()  # accumulation buffer
        expert_outputs = expert_outputs.contiguous()
        indices = token_indices.contiguous().to(torch.int32)

        M = expert_outputs.shape[0]  # number of selected tokens
        N = out.shape[0]             # batch_seq_len
        H = out.shape[1]

        # Launch one program per source row
        grid = (M,)
        BLOCK_SIZE = 128  # moderate chunk size to reduce loop iterations
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, indices, M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,  # small increase in parallelism per program
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
