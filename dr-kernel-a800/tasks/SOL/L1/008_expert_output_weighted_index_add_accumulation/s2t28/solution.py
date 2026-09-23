import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,        # *bf16, shape (M, H), destination
    in_ptr,         # *bf16, shape (N, H), source rows to add
    idx_ptr,        # *int32, shape (N,), token indices (row positions in output)
    M,              # int: number of rows in output (batch_size * seq_len)
    H,              # int: hidden size
    N,              # int: number of selected tokens (rows in expert_outputs)
    BLOCK: tl.constexpr  # columns processed per chunk
):
    # One program per selected token (row in expert_outputs)
    i = tl.program_id(0)
    if i >= N:
        return

    # Load the entire expert row i into a vector across columns
    cols = tl.arange(0, BLOCK)
    in_row_offset = i * H
    in_row_ptrs = in_ptr + in_row_offset + cols
    mask_in = cols < H
    vals = tl.load(in_row_ptrs, mask=mask_in, other=0.0)

    # Destination row index
    dest = tl.load(idx_ptr + i)  # int32
    out_row_ptrs = out_ptr + dest * H + cols

    # Vectorized atomic add for this chunk
    tl.atomic_add(out_row_ptrs, vals, mask=mask_in)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add along dim=0:
          output = final_hidden_states.clone()
          output.index_add_(0, token_indices, expert_outputs)
        """
        # Ensure CUDA tensors for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."
        # Output must be contiguous for linear indexing semantics
        out = final_hidden_states.clone().contiguous()
        in_t = expert_outputs.contiguous()
        idx_t = token_indices.contiguous().to(torch.int32)

        M = out.shape[0]
        N = in_t.shape[0]
        H = in_t.shape[1]

        # Launch one program per row in expert_outputs
        grid = (N,)

        # Fixed, robust configuration that performed best in this environment
        _scatter_add_rows_kernel[grid](
            out, in_t, idx_t,
            M, H, N,
            BLOCK=128,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
