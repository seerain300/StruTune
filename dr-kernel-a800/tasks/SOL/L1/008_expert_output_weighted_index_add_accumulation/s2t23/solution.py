import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,            # *bf16, pointer to output (M, H)
    in_ptr,             # *bf16, pointer to expert_outputs (N, H)
    idx_ptr,            # *int32, pointer to token_indices (N,)
    M: tl.constexpr,    # number of rows in output
    N: tl.constexpr,    # number of rows in expert_outputs
    H: tl.constexpr,    # hidden size
    BLOCK: tl.constexpr # chunk size along hidden dimension
):
    # One Triton program per input row
    i = tl.program_id(0)
    if i >= N:
        return

    # Destination row index
    idx = tl.load(idx_ptr + i)  # int32

    # Vector of column offsets for this chunk
    cols = tl.arange(0, BLOCK)

    # Loop over hidden dimension in chunks
    # Mask handles the tail when H % BLOCK != 0
    for start in range(0, H, BLOCK):
        col = start + cols  # vector of columns
        mask = col < H

        # Compute base pointers for output and input rows
        # out_ptr is (M, H): element address = row * H + col
        out_row_ptr = out_ptr + idx * H
        in_row_ptr = in_ptr + i * H

        # Load current output and expert values for this chunk
        out_vals = tl.load(out_row_ptr + col, mask=mask, other=0.0)
        in_vals = tl.load(in_row_ptr + col, mask=mask, other=0.0)

        # Accumulate
        out_vals += in_vals

        # Store back
        tl.store(out_row_ptr + col, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton execution."

        # Ensure dtypes and contiguity
        out = final_hidden_states.clone()
        assert out.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Expected bfloat16 tensors for outputs and expert_outputs."
        assert token_indices.dtype == torch.long, "token_indices must be torch.long."

        # Make inputs contiguous (should already be, but enforce for safety)
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()

        M = out.shape[0]
        N = expert_outputs.shape[0]
        H = out.shape[1]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)."

        # Cast token indices to int32 for Triton pointer arithmetic
        token_indices_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per row
        BLOCK = 128  # chunk size along hidden dimension, good default
        grid = (N,)  # each program handles one expert row
        _scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices_i32,
            M, N, H, BLOCK,
            num_warps=4, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
