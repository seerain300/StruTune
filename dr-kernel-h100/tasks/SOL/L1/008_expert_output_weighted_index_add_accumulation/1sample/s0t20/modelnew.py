import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,                      # *pointer* to output tensor (bfloat16)
    expert_outputs_ptr,           # *pointer* to expert_outputs tensor (bfloat16)
    token_indices_ptr,            # *pointer* to token_indices tensor (int32)
    stride_out_row: tl.constexpr, # stride along row in out
    stride_out_col: tl.constexpr, # stride along col in out
    stride_exp_row: tl.constexpr, # stride along row in expert_outputs
    stride_exp_col: tl.constexpr, # stride along col in expert_outputs
    batch_seq_len: tl.constexpr,  # number of rows in out
    hidden_size: tl.constexpr,    # number of columns in out
    n_tokens: tl.constexpr,       # number of tokens (experts per token * batch * seq)
    BLOCK_SIZE: tl.constexpr,     # columns per block
):
    # 2D grid: (n_tokens, num_blocks)
    pid_token = tl.program_id(0)
    pid_block = tl.program_id(1)

    # column offsets for this block
    cols = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < hidden_size

    # load token index for this program
    token_idx = tl.load(token_indices_ptr + pid_token)  # int32

    # compute base pointers
    out_row_ptr = out_ptr + token_idx * stride_out_row
    expert_row_ptr = expert_outputs_ptr + pid_token * stride_exp_row

    # masked load of expert vector for this token's block
    vals = tl.load(expert_row_ptr + cols * stride_exp_col, mask=col_mask, other=0.0)

    # masked store into output row at these columns
    tl.store(out_row_ptr + cols * stride_out_col, vals, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        # Shapes derived from inputs
        batch_size = final_hidden_states.shape[0]
        seq_len = final_hidden_states.shape[1]
        hidden_size = final_hidden_states.shape[2]
        batch_seq_len = batch_size * seq_len

        # Prepare output: zero-initialized (GPU) to act as accumulator
        out = torch.zeros(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure inputs are on the same CUDA device and contiguous
        # expert_outputs: (num_selected_tokens, hidden_size), bfloat16
        expert_outputs = expert_outputs.contiguous()
        # token_indices: (num_selected_tokens,), int64 by default from randint, convert to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Dimensions
        num_tokens = token_indices_i32.numel()
        BLOCK_SIZE = 128  # tuneable: 128 or 256 are typical
        num_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)

        # Launch Triton kernel: 2D grid over tokens and column blocks
        grid = (num_tokens, num_blocks)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices_i32,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            batch_seq_len, hidden_size, num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out