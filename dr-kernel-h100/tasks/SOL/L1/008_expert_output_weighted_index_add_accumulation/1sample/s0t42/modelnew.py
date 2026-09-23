import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr, out_stride0, out_stride1,
    src_ptr, src_stride0, src_stride1,
    token_idx_ptr,  # int32 indices
    N_rows, H, N_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D launch: program_id(0) = token index, program_id(1) = column block
    pid_token = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Compute column offsets for this block
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Validity mask: ensure this program corresponds to a valid token and valid column
    token_valid = pid_token < N_tokens
    col_valid = col_offsets < H
    mask = token_valid & col_valid

    # Load token index (safe because we gate with mask; however, we still load with mask)
    tok = tl.load(token_idx_ptr + pid_token, mask=token_valid, other=0)

    # Compute destination row pointer and source pointer for this block
    # out_ptr is [N_rows, H], src_ptr is [N_tokens, H]
    out_row_ptr = out_ptr + tok * out_stride0
    src_row_ptr = src_ptr + pid_token * src_stride0

    # Column pointers
    out_col_ptr = out_row_ptr + col_offsets * out_stride1
    src_col_ptr = src_row_ptr + col_offsets * src_stride1

    # Atomic add for this token's hidden slice
    vals = tl.load(src_col_ptr, mask=mask, other=0.0)
    tl.atomic_add(out_col_ptr, vals, mask=mask)


def run(final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor) -> torch.Tensor:
    """
    Performs atomic accumulation of expert outputs back to token positions using Triton.

    Args:
        final_hidden_states: Accumulation buffer (clone of input), shape [batch_seq_len, hidden_size], dtype bfloat16
        expert_outputs: Weighted outputs from expert computation, shape [num_selected_tokens, hidden_size], dtype bfloat16
        token_indices: Long tensor of original token positions, shape [num_selected_tokens]
    Returns:
        Updated final_hidden_states with expert contributions added
    """
    # Clone to preserve initial random values exactly as in the original PyTorch code
    output = final_hidden_states.clone()

    # Shapes and setup
    batch_seq_len = output.shape[0]
    hidden_size = output.shape[1]
    n_tokens = expert_outputs.shape[0]

    # Triton expects indices as int32 and contiguous
    token_indices_i32 = token_indices.to(torch.int32).contiguous()
    # Ensure source is contiguous
    expert_outputs = expert_outputs.contiguous()

    # Adaptive tiling based on hidden_size
    # Use 1024 column tiles for common and larger hidden sizes
    BLOCK_SIZE = 1024
    num_warps = 8

    # 2D grid over tokens and hidden column blocks
    grid = (n_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))

    # Launch Triton scatter-add kernel
    scatter_add_atomic_kernel[grid](
        output, output.stride(0), output.stride(1),
        expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
        token_indices_i32,
        batch_seq_len, hidden_size, n_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )

    return output


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack tensors in the expected order: final_hidden_states, expert_outputs, token_indices
        if len(args) != 3:
            raise RuntimeError("ModelNew.forward expects three tensors: final_hidden_states, expert_outputs, token_indices.")
        final_hidden_states, expert_outputs, token_indices = args
        return run(final_hidden_states, expert_outputs, token_indices)