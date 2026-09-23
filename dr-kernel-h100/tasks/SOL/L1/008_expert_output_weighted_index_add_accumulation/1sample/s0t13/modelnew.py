import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add one program per token.
# It loops over hidden columns in chunks and performs atomic_add into the target row.
@triton.jit
def scatter_add_tokens_kernel(
    out_ptr,              # *bf16
    expert_ptr,           # *bf16
    token_idx_ptr,        # *i32
    batch_seq_len,        # int
    hidden_size,          # int
    n_tokens,             # int
    stride_row,           # int (elements)
    stride_col,           # int (elements)
    exp_stride_row,       # int (elements)
    exp_stride_col,       # int (elements)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # token id
    # Guard: if pid >= n_tokens, early return (shouldn't happen with grid=(n_tokens,))
    if pid >= n_tokens:
        return

    # Load token index for this program (int32)
    token_index = tl.load(token_idx_ptr + pid)

    # Iterate over hidden columns in chunks
    col = 0
    while col < hidden_size:
        offs = tl.arange(0, BLOCK_SIZE)
        col_ids = col + offs
        mask = col_ids < hidden_size

        # Compute pointers
        # out_ptr is (batch_seq_len, hidden_size) with strides (stride_row, stride_col)
        out_ptrs = out_ptr + token_index * stride_row + col_ids * stride_col
        # expert_ptr is (n_tokens, hidden_size) with strides (exp_stride_row, exp_stride_col)
        exp_ptrs = expert_ptr + pid * exp_stride_row + col_ids * exp_stride_col

        # Load expert slice for this token; masked for tail
        vals = tl.load(exp_ptrs, mask=mask, other=0.0)

        # Atomic add into the output
        tl.atomic_add(out_ptrs, vals, mask=mask)

        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        # Shapes and parameters
        batch_size, seq_len, hidden_size = (
            0,  # not used directly, hidden_size comes from axes_and_scalars in get_inputs
            0,  # not used directly
            0,  # assume get_inputs sets hidden_size correctly
        )
        # We don't have batch_size/seq_len in the args; they are provided via axes_and_scalars in get_inputs.
        # However, final_hidden_states shape is (batch_seq_len, hidden_size). So we use its shape for output.
        # But since forward only receives final_hidden_states, expert_outputs, token_indices, we infer batch_seq_len
        # from the device and shape of final_hidden_states.
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Ensure output is zero-initialized
        out = torch.zeros((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_row = out.stride(0)
        stride_col = out.stride(1)
        exp_stride_row = expert_outputs.stride(0)
        exp_stride_col = expert_outputs.stride(1)

        # Launch Triton kernel: one program per token
        if TRITON_AVAILABLE:
            grid = (n_tokens,)
            scatter_add_tokens_kernel[grid](
                out, expert_outputs, token_indices_i32,
                batch_seq_len, hidden_size, n_tokens,
                stride_row, stride_col,
                exp_stride_row, exp_stride_col,
                BLOCK_SIZE=128,  # chunk size for hidden columns; tune as needed
                num_warps=4,
            )
        else:
            # Fallback: PyTorch implementation if Triton is not available
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return out