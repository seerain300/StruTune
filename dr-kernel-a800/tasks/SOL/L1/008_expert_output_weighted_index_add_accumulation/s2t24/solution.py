import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,       # *bf16, shape (M, H), output buffer
    in_ptr,        # *bf16, shape (N, H), expert outputs
    idx_ptr,       # *int32, shape (N,), token indices
    M,             # int: number of rows in output (i.e., batch_size * seq_len)
    H: tl.constexpr,  # int: hidden size (columns)
    BLOCK: tl.constexpr  # int: block size across columns
):
    # One program per row i in the source (expert_outputs)
    i = tl.program_id(0)
    # We launch exactly N programs, where N is the number of rows in 'in_ptr'
    # (i.e., num_selected_tokens). So no extra bounds check is needed.
    # Load destination index
    idx = tl.load(idx_ptr + i)  # int32
    # Compute base pointers for this row
    out_row_base = out_ptr + idx * H
    in_row_base = in_ptr + i * H

    # Process the hidden dimension in chunks of BLOCK columns
    # This loop uses Triton's compile-time unrolled pattern via range(0, H, BLOCK).
    # Triton requires the loop bounds to be known at compile time, so we pass H as tl.constexpr.
    for col_start in range(0, H, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)  # vector of column offsets
        mask = offs < H                         # mask for tail
        # Load a chunk of the source row
        vals = tl.load(in_row_base + offs, mask=mask, other=0.0)
        # Atomic add the chunk into the destination row
        tl.atomic_add(out_row_base + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure we have CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton"
        # Clone to preserve initial random values exactly
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        out = output  # already contiguous as clone of final_hidden_states
        in_t = expert_outputs.contiguous()  # ensure contiguous for simple pointer math
        # Triton expects int32 indices
        idx_t = token_indices.to(torch.int32)

        M = out.shape[0]           # number of rows in output (batch_seq_len)
        N = in_t.shape[0]          # number of selected tokens (num_selected_tokens)
        H = in_t.shape[1]          # hidden size

        # Launch one program per row in 'in_t'
        grid = (N,)
        # Choose a fixed BLOCK to avoid autotune variability
        BLOCK = 128
        _scatter_add_rows_kernel[grid](
            out, in_t, idx_t,
            M, H, BLOCK,
            num_warps=4, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
