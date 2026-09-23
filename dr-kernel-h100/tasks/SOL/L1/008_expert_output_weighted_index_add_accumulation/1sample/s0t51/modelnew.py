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

    # Load token index for this program (row destination)
    # Note: token_idx_ptr is int32
    row_idx = tl.load(token_idx_ptr + pid_token)

    # Compute linear offsets for out and src slices
    out_offsets = row_idx * out_stride0 + col_offsets * out_stride1
    src_offsets = pid_token * src_stride0 + col_offsets * src_stride1

    # Load source slice and perform atomic add into output
    src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + out_offsets, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Performs atomic accumulation of expert outputs back to token positions.
        final_hidden_states: (batch_seq_len, hidden_size), bfloat16, random init
        expert_outputs: (num_selected_tokens, hidden_size), bfloat16
        token_indices: (num_selected_tokens,), long
        Returns updated final_hidden_states with expert contributions added.
        """
        # Preserve initial random values by cloning
        output = final_hidden_states.clone()

        # Ensure device and contiguity; token_indices must be on same device
        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        batch_size = final_hidden_states.shape[0]
        batch_seq_len = batch_size * final_hidden_states.shape[1]  # original code uses batch_size * seq_len
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Choose adaptive tiling
        if hidden_size <= 256:
            BLOCK_SIZE = 1024
            num_warps = 8
        elif hidden_size <= 2048:
            BLOCK_SIZE = 1024
            num_warps = 8
        else:
            BLOCK_SIZE = 2048  # larger tile to reduce grid size for very large hidden sizes
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