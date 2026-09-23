import torch
import triton
import triton.language as tl


@triton.jit
def per_token_scatter_add_kernel(
    out_ptr,              # *bf16, (B, H)
    expert_ptr,           # *bf16, (N, H)
    token_idx_ptr,        # *int32, (N,)
    out_stride0, out_stride1,
    expert_stride0, expert_stride1,
    B, H, N,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index
    row = tl.load(token_idx_ptr + pid)  # int32

    # Iterate over columns in chunks
    for col in range(0, H, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Load the expert vector chunk for this token
        # out[row, cols] += expert[pid, cols]
        exp_ptrs = expert_ptr + pid * expert_stride0 + cols * expert_stride1
        exp_vec = tl.load(exp_ptrs, mask=mask, other=0.0)  # bf16

        out_ptrs = out_ptr + row * out_stride0 + cols * out_stride1
        out_vec = tl.load(out_ptrs, mask=mask, other=0.0)  # bf16

        # Add and store back
        out_vec = out_vec + exp_vec
        tl.store(out_ptrs, out_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-based implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Clone to match original behavior (preserve initial random values)
        out = final_hidden_states.clone()
        # Ensure tensors are contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Shapes
        B = out.shape[0]         # batch_seq_len
        H = out.shape[1]         # hidden_size
        N = token_indices.numel()  # num_selected_tokens

        # Triton expects int32 indices for pointer math
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose column chunk size
        BLOCK_SIZE = 128  # tuneable: 128 or 256

        # Launch one program per token
        grid = (N,)
        per_token_scatter_add_kernel[grid](
            out, expert_outputs, token_indices_i32,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            B, H, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune: 4-8 are typical
        )

        return out