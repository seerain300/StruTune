import torch
import triton
import triton.language as tl


@triton.jit
def zero_init_kernel(out_ptr, stride_out_row, stride_out_col, H, W, BLOCK_SIZE: tl.constexpr):
    # Zero-initialize out_ptr of shape (H, W) using 2D tiling
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < H) & (cols < W)
    # out_ptr is a 2D pointer; we compute addresses as row-major: row * stride_row + cols * stride_col
    out_ptrs = out_ptr + row * stride_out_row + cols * stride_out_col
    # Write zeros
    tl.store(out_ptrs, 0.0, mask=mask)


@triton.jit
def scatter_add_kernel(out_ptr, stride_out_row, stride_out_col,
                       expert_ptr, stride_exp_row, stride_exp_col,
                       token_indices_ptr,
                       H, W, N,
                       BLOCK_SIZE: tl.constexpr):
    # One program per token
    token = tl.program_id(0)
    # Bounds check on token (not strictly necessary if grid=(N,))
    if token >= N:
        return

    # Load token index (row destination)
    row_dst = tl.load(token_indices_ptr + token)

    # Iterate over columns in BLOCK_SIZE chunks
    # We use a simple while loop to cover all columns
    col_start = 0
    while col_start < W:
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < W

        # Load the corresponding slice of expert_outputs[token, :]
        src_ptrs = expert_ptr + token * stride_exp_row + cols * stride_exp_col
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        # Compute destination pointers in out
        dst_ptrs = out_ptr + row_dst * stride_out_row + cols * stride_out_col

        # Store the loaded values (direct copy) into out at row_dst
        tl.store(dst_ptrs, vals, mask=mask)

        col_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output.index_add(dim=0, index=token_indices, source=expert_outputs)
        where:
            - final_hidden_states is the accumulation buffer of shape (batch_seq_len, hidden_size),
              dtype bfloat16. We will not mutate it; we create output.
            - expert_outputs is (num_selected_tokens, hidden_size), dtype bfloat16.
            - token_indices is (num_selected_tokens,), dtype long (int64), values in [0, batch_seq_len).
        """
        # Extract shapes
        H = final_hidden_states.shape[0]
        W = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Output accumulator: start from zeros to match index_add semantics
        out = torch.zeros((H, W), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure expert_outputs is contiguous and dtype matches
        expert_outputs = expert_outputs.contiguous()

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose a column block size
        BLOCK_SIZE = 256 if W >= 256 else 128

        # Launch zero-init kernel: 2D grid over rows and column blocks
        grid_zero = (H, triton.cdiv(W, BLOCK_SIZE))
        zero_init_kernel[grid_zero](
            out, out.stride(0), out.stride(1), H, W, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Launch scatter-add kernel: one program per token
        grid_scatter = (N,)
        scatter_add_kernel[grid_scatter](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            H, W, N,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        return out