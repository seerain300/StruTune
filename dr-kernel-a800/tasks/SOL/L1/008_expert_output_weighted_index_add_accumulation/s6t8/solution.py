import torch
import triton
import triton.language as tl


@triton.jit
def _row_copy_kernel(
    output_ptr,  # *const T
    src_ptr,     # *const T
    rows,        # int32 (M = batch_seq_len)
    hidden_size, # int32
):
    row_id = tl.program_id(axis=0)
    if row_id >= rows:
        return
    # Compute base offsets
    out_row_offset = row_id * hidden_size
    src_row_offset = row_id * hidden_size
    # Vectorized copy across hidden dimension
    offs = tl.arange(0, hidden_size)
    out_ptrs = output_ptr + out_row_offset + offs
    src_ptrs = src_ptr + src_row_offset + offs
    vals = tl.load(src_ptrs)
    tl.store(out_ptrs, vals)


@triton.jit
def _scatter_add_row_kernel(
    output_ptr,       # *T (already initialized by _row_copy_kernel)
    expert_ptr,       # *const T (expert_outputs)
    token_indices_ptr,# *const int32 (indices)
    num_tokens,       # int32 (num_selected_tokens)
    hidden_size,      # int32
):
    token_id = tl.program_id(axis=0)
    if token_id >= num_tokens:
        return
    # Load index (int32)
    row_index = tl.load(token_indices_ptr + token_id)
    # Compute base offsets
    out_row_offset = row_index * hidden_size
    exp_row_offset = token_id * hidden_size
    # Vectorized load and add
    offs = tl.arange(0, hidden_size)
    out_ptrs = output_ptr + out_row_offset + offs
    exp_ptrs = expert_ptr + exp_row_offset + offs
    vals = tl.load(exp_ptrs)
    current = tl.load(out_ptrs)  # read current row from output
    new_vals = current + vals
    tl.store(out_ptrs, new_vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:

        # Output must be a clone of final_hidden_states; we do the copy via Triton
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]

        # Ensure tensors are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        # Allocate output as empty (we will fill it via Triton copy)
        output = torch.empty((batch_seq_len, hidden_size), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Cast token indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        # Number of tokens
        num_selected_tokens = expert_outputs.shape[0]

        # 1) Copy final_hidden_states into output (Triton kernel)
        grid_copy = (batch_seq_len,)
        _row_copy_kernel[grid_copy](
            output, final_hidden_states,
            batch_seq_len, hidden_size,
            num_warps=4, num_stages=2
        )

        # 2) Scatter-add via Triton: output[token_indices[i]] += expert_outputs[i]
        grid_add = (num_selected_tokens,)
        _scatter_add_row_kernel[grid_add](
            output, expert_outputs, token_indices,
            num_selected_tokens, hidden_size,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
