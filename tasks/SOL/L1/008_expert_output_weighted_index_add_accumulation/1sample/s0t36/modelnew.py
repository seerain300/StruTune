import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr, out_stride0, out_stride1,
    src_ptr, src_stride0, src_stride1,
    token_idx_ptr,  # int32
    N_rows, H, N_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid_token = tl.program_id(0)  # token id
    pid_col = tl.program_id(1)    # column block id
    # Compute column offsets for this block
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < H

    # Load token index for this program
    tok = tl.load(token_idx_ptr + pid_token)  # int32
    # Compute pointers for out and src for this token and column block
    out_ptrs = out_ptr + tok * out_stride0 + col_offsets * out_stride1
    src_ptrs = src_ptr + pid_token * src_stride0 + col_offsets * src_stride1

    # Atomic add: add src slice into out row at tok
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Implemented as Triton scatter-add with atomic accumulation.
        """
        # Preserve initial random values
        output = final_hidden_states.clone()

        # Shapes and setup
        batch_seq_len = output.shape[0]
        hidden_size = output.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Triton expects indices as int32 and contiguous
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        # Ensure source is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Choose tile size and warps based on hidden_size
        if hidden_size <= 128:
            BLOCK_SIZE = 256
            num_warps = 4
        elif hidden_size <= 512:
            BLOCK_SIZE = 512
            num_warps = 8
        else:
            BLOCK_SIZE = 256
            num_warps = 4

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