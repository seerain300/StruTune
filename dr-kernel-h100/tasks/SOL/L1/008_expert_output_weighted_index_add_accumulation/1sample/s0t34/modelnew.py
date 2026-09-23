import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,                 # *bf16
    out_stride_row, out_stride_col,  # int32 strides
    expert_ptr,              # *bf16
    expert_stride_row, expert_stride_col,  # int32 strides
    token_indices_ptr,       # *int32
    batch_seq_len, hidden_size, n_tokens,  # int32 scalars
    BLOCK_SIZE: tl.constexpr,
):
    # program ids
    pid_tok = tl.program_id(0)  # token index for this program
    pid_col = tl.program_id(1)  # block id along hidden columns

    # Compute column offsets for this block
    col_start = pid_col * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_size

    # Load token index for this program (int32)
    token_idx = tl.load(token_indices_ptr + pid_tok)

    # Compute base pointers
    # out: row = token_idx, cols = col_offsets
    out_row_ptr = out_ptr + token_idx * out_stride_row
    out_row_cols_ptr = out_row_ptr + col_offsets * out_stride_col

    # expert: row = pid_tok, cols = col_offsets
    expert_row_ptr = expert_ptr + pid_tok * expert_stride_row
    expert_row_cols_ptr = expert_row_ptr + col_offsets * expert_stride_col

    # Load expert slice and perform atomic add
    expert_slice = tl.load(expert_row_cols_ptr, mask=mask, other=0.0)
    out_slice = tl.load(out_row_cols_ptr, mask=mask, other=0.0)
    out_slice += expert_slice
    tl.atomic_add(out_row_cols_ptr, expert_slice, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Returns the updated output.
        """
        # Ensure inputs are on CUDA and contiguous; Triton requires CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"

        # Clone to preserve initial random values (PyTorch behavior)
        output = final_hidden_states.clone()

        # Shapes
        batch_seq_len = output.shape[0]
        hidden_size = output.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Triton expects indices as int32
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Ensure expert tensor is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Choose block size and grid
        BLOCK_SIZE = 256  # robust default; 128/256 work well for hidden sizes up to 1024
        grid = (n_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))

        # Launch scatter-add kernel (atomic accumulation into output)
        scatter_add_atomic_kernel[grid](
            output, output.stride(0), output.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return output