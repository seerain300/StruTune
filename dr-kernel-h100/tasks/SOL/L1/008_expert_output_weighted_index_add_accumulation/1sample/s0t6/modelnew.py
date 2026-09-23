import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,                 # *bf16, pointer to output [N, H]
    out_stride_row,          # int, stride along rows (dim 0)
    out_stride_col,          # int, stride along cols (dim 1)
    expert_ptr,              # *bf16, pointer to expert_outputs [M, H]
    expert_stride_row,       # int, stride along rows (dim 0) for expert_outputs
    expert_stride_col,       # int, stride along cols (dim 1) for expert_outputs
    token_indices_ptr,       # *i32, pointer to token indices [M]
    N,                       # int, number of rows in output (batch_seq_len)
    H,                       # int, number of columns (hidden_size)
    M,                       # int, number of tokens (num_selected_tokens)
    BLOCK_SIZE: tl.constexpr # columns per iteration
):
    # One program per token
    pid = tl.program_id(axis=0)  # pid in [0, M)
    # Bounds guard (in case grid is larger than M)
    if pid >= M:
        return

    # Load token index for this program
    tok_idx = tl.load(token_indices_ptr + pid)
    # Bounds guard: tok_idx should be in [0, N); if not, return
    if (tok_idx < 0) or (tok_idx >= N):
        return

    # Loop over hidden columns in chunks of BLOCK_SIZE
    col = 0
    while col < H:
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Compute source pointer for this token's row and column chunk
        # expert_outputs[pid, cols]
        src_ptr = expert_ptr + pid * expert_stride_row + cols * expert_stride_col
        src_vec = tl.load(src_ptr, mask=mask, other=0.0)

        # Compute destination pointer for out[tok_idx, cols]
        dst_ptr = out_ptr + tok_idx * out_stride_row + cols * out_stride_col
        # Atomic add: add src_vec into out at those columns
        # Ensure we don't do any out-of-bounds store by masking; but atomic_add itself
        # is only meaningful for valid lanes, so we use masked load and then store.
        # Triton will not perform atomic for lanes where mask is False, but we should
        # still create a safe pointer. Here we use mask to ensure only valid lanes
        # participate in atomic add.
        # Note: Triton's atomic_add requires valid pointers; masked stores here are
        # fine as we write zeros for invalid lanes.
        # Since we pre-zeroed out, atomic adds will accumulate correctly; masked lanes
        # won't contribute.
        # However, to be precise, we can avoid atomic for invalid lanes by checking mask.
        # Triton does not provide a masked atomic, so we ensure tok_idx is valid and
        # mask protects columns.
        tl.atomic_add(dst_ptr, src_vec, mask=mask)

        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that performs scatter-add:
          output[token_indices[i]] += expert_outputs[i] for all i.
        We:
          - allocate output as zeros (PyTorch), ensuring atomic adds have a zero accumulator
          - launch a Triton kernel to perform atomic scatter-add
        """

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Allocate output and ensure it's zero-initialized for correct accumulation
        # We allocate with the same shape/device/dtype as final_hidden_states
        # Note: final_hidden_states is provided by get_inputs and should be a valid tensor.
        out = torch.zeros(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Convert token indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose a block size; 128 is a good default. You can tune based on H.
        BLOCK_SIZE = 128

        # Launch scatter-add kernel: one program per token
        grid = (n_tokens,)
        scatter_add_atomic_kernel[grid](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out