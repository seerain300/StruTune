import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,  # *bf16, shape [batch_seq_len, hidden_size]
    out_stride_row, out_stride_col,  # strides for out
    src_ptr,  # *bf16, shape [n_tokens, hidden_size]
    src_stride_row, src_stride_col,  # strides for src (expert_outputs)
    token_indices_ptr,  # *i32, shape [n_tokens]
    batch_seq_len: tl.int32,        # number of rows in out
    hidden_size: tl.int32,          # number of columns in out/src
    n_tokens: tl.int32,             # number of tokens
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # token id
    # Guard against out-of-range pids (in case grid > n_tokens)
    if pid >= n_tokens:
        return

    # Load token index for this program
    token_index = tl.load(token_indices_ptr + pid)
    # If token_index is out of range, skip (defensive; grid should match n_tokens)
    if token_index < 0 or token_index >= batch_seq_len:
        return

    # Iterate over hidden columns in chunks of BLOCK_SIZE
    col_start = 0
    while col_start < hidden_size:
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Compute pointers for src row pid and out row token_index
        src_row_base = src_ptr + pid * src_stride_row
        out_row_ptr = out_ptr + token_index * out_stride_row

        # Load expert vector slice for this token
        src_ptrs = src_row_base + cols * src_stride_col
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        # Store into output with atomic add
        out_ptrs = out_row_ptr + cols * out_stride_col
        tl.atomic_add(out_ptrs, vals, mask=mask)

        col_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        """
        Triton-only implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We initialize output to zeros using torch.zeros (reliable), then use Triton to perform atomic scatter-add.
        """
        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Ensure dtype is bfloat16 and device is CUDA (Triton requires CUDA)
        # Output must be zero-initialized to match index_add semantics
        out = torch.zeros(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Make sure inputs are contiguous for simpler stride handling
        out = out  # already allocated
        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose a block size; 128 is a good default for typical hidden sizes
        BLOCK_SIZE = 128

        # Launch scatter-add kernel: one program per token
        grid = (n_tokens,)
        scatter_add_atomic_kernel[grid](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out