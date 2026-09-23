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
    # 2D launch: each program handles one token and one column block
    pid_token = tl.program_id(0)  # token id
    pid_col = tl.program_id(1)    # column block id

    # Compute column offsets for this block
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < H

    # Load token index (int32)
    tok = tl.load(token_idx_ptr + pid_token)  # scalar int32

    # Compute per-row and per-column strides
    # out_ptr points to row 0, col 0. Address of out[tok, col_offsets]:
    out_row_base = tok * out_stride0
    src_row_base = pid_token * src_stride0  # pid_token is token index into src rows

    # Form pointers for this tile
    out_ptrs = out_ptr + out_row_base + col_offsets * out_stride1
    src_ptrs = src_ptr + src_row_base + col_offsets * src_stride1

    # Atomic add: add this tile of expert_outputs to the corresponding positions in output
    # Use 'other=0' in load to ignore invalid lanes, but atomic_add only affects valid lanes via mask.
    # Triton supports atomic_add for fp16/bf16/fp32.
    tl.atomic_add(out_ptrs, tl.load(src_ptrs, mask=col_mask, other=0.0), mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add:
        output = final_hidden_states.clone()
        output[token_indices[i]] += expert_outputs[i] for all i
        """
        # Preserve initial random values
        output = final_hidden_states.clone()

        # Shapes
        batch_seq_len = output.shape[0]
        hidden_size = output.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Triton expects indices as int32 and contiguous
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        # Ensure source tensor is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Adaptive tiling based on hidden_size
        if hidden_size <= 256:
            BLOCK_SIZE = 512
            num_warps = 8
        elif hidden_size <= 1024:
            BLOCK_SIZE = 1024
            num_warps = 8
        else:
            BLOCK_SIZE = 512
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