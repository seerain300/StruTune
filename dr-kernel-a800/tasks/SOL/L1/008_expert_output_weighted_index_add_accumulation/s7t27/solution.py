import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,        # *bf16, shape (N, H), contiguous
    src_ptr,        # *bf16, shape (M, H), contiguous
    indices_ptr,    # *int32, shape (M,)
    M,              # int32, number of rows in src (num_selected_tokens)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # compile-time block size for unrolling
):
    # One program per row i
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index for this expert output
    dst_row = tl.load(indices_ptr + pid)  # int32

    # Loop over hidden dimension in chunks of BLOCK_SIZE, unrolled
    for off in tl.static_range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Compute pointers for this row (row-major contiguous)
        out_row_ptr = out_ptr + dst_row * H + cols
        src_row_ptr = src_ptr + pid * H + cols

        # Load a block of values from src and add atomically to out
        vals = tl.load(src_row_ptr, mask=mask, other=0.0)  # *bf16
        tl.atomic_add(out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match reference behavior
        out = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        assert expert_outputs.is_cuda and out.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert expert_outputs.dtype == torch.bfloat16 and out.dtype == torch.bfloat16, "dtype must be bfloat16"
        assert expert_outputs.is_contiguous() and out.is_contiguous(), "Tensors must be contiguous"

        # Convert indices to int32 for Triton
        indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per row
        M = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose a block size suitable for typical hidden sizes; 128 works well
        BLOCK_SIZE = 128

        grid = (M,)
        scatter_add_per_row_kernel[grid](
            out, expert_outputs, indices,
            M, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
