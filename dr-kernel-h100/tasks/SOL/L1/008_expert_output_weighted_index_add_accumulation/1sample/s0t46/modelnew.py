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
    token = tl.load(token_idx_ptr + pid_token)
    # Compute destination row pointer
    dest_row_ptr = out_ptr + token * out_stride0

    # Load source slice for this token and column block
    src_row_ptr = src_ptr + pid_token * src_stride0
    src_vals = tl.load(src_row_ptr + col_offsets * src_stride1, mask=mask, other=0.0)

    # Atomic add into destination row
    tl.atomic_add(dest_row_ptr + col_offsets * out_stride1, src_vals, mask=mask)


def triton_scatter_add(final_hidden_states: torch.Tensor,
                       expert_outputs: torch.Tensor,
                       token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized scatter-add: output[token_indices[i]] += expert_outputs[i]
    Preserves initial final_hidden_states by cloning.
    """
    # Ensure device and contiguity
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA."
    output = final_hidden_states.clone().contiguous()
    expert_outputs = expert_outputs.contiguous()
    token_indices_i32 = token_indices.to(torch.int32).contiguous()

    batch_seq_len, hidden_size = output.shape
    n_tokens, H = expert_outputs.shape
    assert H == hidden_size, "expert_outputs hidden_size must match output hidden_size"

    # Heuristic tiling: use larger block to reduce column tiles for common sizes
    if hidden_size <= 256:
        BLOCK_SIZE = 1024
        num_warps = 8
    elif hidden_size <= 1024:
        BLOCK_SIZE = 1024
        num_warps = 8
    elif hidden_size <= 2048:
        BLOCK_SIZE = 2048
        num_warps = 8
    else:
        BLOCK_SIZE = 2048
        num_warps = 8

    grid = (n_tokens, triton.cdiv(hidden_size, BLOCK_SIZE))

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
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        return triton_scatter_add(final_hidden_states, expert_outputs, token_indices)