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
    mask = col_offsets < H

    # Load token index for this program
    tok = tl.load(token_idx_ptr + pid_token)

    # Compute base pointers for destination row and source row
    # Addressing uses strides in elements (Triton will handle element-wise indexing)
    out_row_ptrs = out_ptr + tok * out_stride0 + col_offsets * out_stride1
    src_row_ptrs = src_ptr + pid_token * src_stride0 + col_offsets * src_stride1

    # Atomic add for masked columns
    vals = tl.load(src_row_ptrs, mask=mask, other=0.0)
    tl.atomic_add(out_row_ptrs, vals, mask=mask)


def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs atomic accumulation of expert outputs back to token positions.
    Args:
        final_hidden_states: Accumulation buffer for all tokens (batch_seq_len, hidden_size)
        expert_outputs: Weighted outputs from expert computation (num_selected_tokens, hidden_size)
        token_indices: Original token positions (num_selected_tokens,)
    Returns:
        Updated final_hidden_states with expert contributions added
    """
    # Clone to avoid modifying input in-place for correctness
    output = final_hidden_states.clone()

    # Shapes
    batch_seq_len = output.shape[0]
    hidden_size = output.shape[1]
    n_tokens = expert_outputs.shape[0]

    # Ensure contiguous and dtype setup
    expert_outputs = expert_outputs.contiguous()
    # Triton expects indices as int32 for efficient addressing
    token_indices_i32 = token_indices.to(torch.int32).contiguous()

    # Adaptive tiling based on hidden_size
    # Using 1024 columns per tile gives good performance across a wide range.
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
        # Expect the same argument order as the original: (final_hidden_states, expert_outputs, token_indices)
        # Model.forward in the original code takes *args and unpacks them; we mirror that here.
        if len(args) != 3:
            raise RuntimeError("ModelNew.forward expects three tensors: final_hidden_states, expert_outputs, token_indices.")
        final_hidden_states, expert_outputs, token_indices = args
        return run(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
