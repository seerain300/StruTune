import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel_1d(
    out_ptr, out_stride0, out_stride1,
    src_ptr, src_stride0, src_stride1,
    token_idx_ptr,  # int32 indices
    N_rows, H, N_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D launch: one program per token
    pid = tl.program_id(0)

    # Guard in case grid > N_tokens (shouldn't happen if we set grid=N_tokens)
    if pid >= N_tokens:
        return

    # Load token index (row destination)
    tok = tl.load(token_idx_ptr + pid)
    # Triton allows int32 indices; ensure tok in range [0, N_rows)
    # (We assume token_indices are valid per the original contract.)

    # Iterate over hidden columns in chunks of BLOCK_SIZE
    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Compute pointers for this slice
        out_ptrs = out_ptr + tok * out_stride0 + cols * out_stride1
        src_ptrs = src_ptr + pid * src_stride0 + cols * src_stride1

        # Load source slice and atomically add to destination row
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        tl.atomic_add(out_ptrs, vals, mask=mask)


def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs atomic accumulation of expert outputs back to token positions using a Triton kernel.

    Args:
        final_hidden_states: Accumulation buffer for all tokens (batch_seq_len, hidden_size), bfloat16.
        expert_outputs: Weighted outputs from expert computation (num_selected_tokens, hidden_size), bfloat16.
        token_indices: Original token positions (num_selected_tokens,), int64 (will be converted to int32).
        
    Returns:
        Updated final_hidden_states with expert contributions added.
    """
    # Preserve initial random values exactly like the original
    output = final_hidden_states.clone()

    # Shapes
    batch_seq_len = output.shape[0]
    hidden_size = output.shape[1]
    n_tokens = expert_outputs.shape[0]

    # Ensure inputs are contiguous and dtypes are appropriate
    expert_outputs = expert_outputs.contiguous()
    token_indices_i32 = token_indices.to(torch.int32).contiguous()

    # Kernel launch configuration
    BLOCK_SIZE = 256  # safe chunk size; tune if needed
    num_warps = 4     # conservative choice for occupancy

    # 1D grid over tokens
    grid = (n_tokens,)

    # Launch Triton kernel
    scatter_add_atomic_kernel_1d[grid](
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
        if len(args) != 3:
            raise RuntimeError("ModelNew.forward expects three tensors: final_hidden_states, expert_outputs, token_indices.")
        final_hidden_states, expert_outputs, token_indices = args
        return run(final_hidden_states, expert_outputs, token_indices)