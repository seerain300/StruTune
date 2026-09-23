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
    token_idx = tl.load(token_idx_ptr + pid_token)

    # Compute base pointers for destination row and source slice
    out_row_ptr = out_ptr + token_idx * out_stride0
    src_row_ptr = src_ptr + pid_token * src_stride0

    # Load source values for this block
    src_vals = tl.load(src_row_ptr + col_offsets * src_stride1, mask=mask, other=0.0)

    # Atomic add into destination row
    tl.atomic_add(out_row_ptr + col_offsets * out_stride1, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Performs scatter-add: output = final_hidden_states.clone(); then
        output[token_indices[i]] += expert_outputs[i] for each token i.
        The accumulation is done via a Triton kernel using atomic adds.
        """

        # Preserve initial random values exactly as in the original
        output = final_hidden_states.clone()

        # Ensure contiguity and proper dtypes for Triton
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Adaptive tiling based on hidden_size
        if hidden_size <= 256:
            BLOCK_SIZE = 1024
            num_warps = 8
        elif hidden_size <= 4096:
            BLOCK_SIZE = 2048
            num_warps = 8
        else:
            BLOCK_SIZE = 2048
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


def run(*args):
    return ModelNew()(*args)
