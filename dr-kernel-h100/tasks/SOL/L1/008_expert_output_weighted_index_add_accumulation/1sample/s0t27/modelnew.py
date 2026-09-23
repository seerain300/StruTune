import torch
import triton
import triton.language as tl


@triton.jit
def row_accumulator_atomic(
    out_ptr,  # *bf16
    expert_ptr,  # *bf16
    token_idx_ptr,  # *int32
    batch_size, seq_len, hidden_size, n_tokens,
    out_stride_row, out_stride_col,
    exp_stride_row, exp_stride_col,
    BLOCK_SIZE: tl.constexpr,     # columns per block
    BLOCK_TOK: tl.constexpr       # tokens per block
):
    # One program per row
    r = tl.program_id(0)
    # If grid covers more rows than batch_seq_len, guard with mask; but grid should equal batch_seq_len
    # Compute column offsets
    cols = tl.arange(0, BLOCK_SIZE)
    # Loop over tokens in blocks
    tok = 0
    while tok < n_tokens:
        tok_offsets = tok + tl.arange(0, BLOCK_TOK)
        mask_tok = tok_offsets < n_tokens
        # Load token indices for this block
        token_ids = tl.load(token_idx_ptr + tok_offsets, mask=mask_tok, other=0)

        # Identify tokens that target this row
        in_row = token_ids == r  # boolean per token in block
        in_row = in_row & mask_tok  # ensure valid tokens

        # Iterate over this block to atomic_add each contributing token's vector
        # We'll build a per-column vector to accumulate contributions for this row
        add_vec = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)

        # For each token j in the block that targets row r, add expert[j, :]
        # We loop j in [0, BLOCK_TOK) using masking
        for j in range(BLOCK_TOK):
            # If this token is in-row
            if in_row[j]:
                # Compute expert row j vector for columns [0, BLOCK_SIZE)
                col = cols  # vector
                col_mask = col < hidden_size
                # Load expert row j for these columns
                exp_row_ptr = expert_ptr + j * exp_stride_row
                exp_vec = tl.load(exp_row_ptr + col * exp_stride_col, mask=col_mask, other=0.0)
                add_vec += exp_vec

        # Add to the corresponding output row
        out_row_ptr = out_ptr + r * out_stride_row
        out_vec = tl.load(out_row_ptr + cols * out_stride_col, mask=(cols < hidden_size), other=0.0)
        out_vec += add_vec
        tl.store(out_row_ptr + cols * out_stride_col, out_vec, mask=(cols < hidden_size))

        tok += BLOCK_TOK


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that matches the original index_add behavior.
        - out starts as a clone of final_hidden_states to preserve its initial random values.
        - Triton kernel processes each row, atomically adding contributions from all tokens that target it.
        """
        # Ensure contiguous and device/dtype
        out = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()
        device = out.device

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        batch_size = final_hidden_states.shape[0]
        seq_len = final_hidden_states.shape[1]
        hidden_size = final_hidden_states.shape[2]
        n_tokens = token_indices_i32.numel()

        # Grid: one program per row
        grid = (batch_size * seq_len,)

        # Launch Triton kernel
        row_accumulator_atomic[grid](
            out, expert_outputs, token_indices_i32,
            batch_size, seq_len, hidden_size, n_tokens,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            BLOCK_SIZE=128,     # columns per block
            BLOCK_TOK=64,       # tokens per block
            num_warps=4,        # tune as needed
        )

        return out