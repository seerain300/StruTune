import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,                # *bf16, [rows, cols] = [batch_seq_len, hidden_size]
    out_stride_row,         # int
    out_stride_col,         # int
    expert_ptr,             # *bf16, [n_tokens, hidden_size]
    expert_stride_row,      # int
    expert_stride_col,      # int
    token_indices_ptr,      # *int32, [n_tokens]
    batch_seq_len,          # int
    hidden_size,            # int
    n_tokens,               # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Guard: if pid >= n_tokens, do nothing (safety in case grid > n_tokens)
    if pid >= n_tokens:
        return

    # Load token index for this program
    token_index = tl.load(token_indices_ptr + pid)
    # If token_index is out of bounds, skip (robustness; in this task indices are valid)
    if token_index < 0 or token_index >= batch_seq_len:
        return

    # Iterate over hidden_size in BLOCK_SIZE chunks
    col_start = 0
    while col_start < hidden_size:
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load the corresponding slice of expert_outputs[pid, :]
        vals = tl.load(
            expert_ptr + pid * expert_stride_row + cols * expert_stride_col,
            mask=mask,
            other=0.0,
        )

        # Compute destination pointers for out[token_index, cols]
        out_ptrs = out_ptr + token_index * out_stride_row + cols * out_stride_col

        # Atomic add into output
        tl.atomic_add(out_ptrs, vals, mask=mask)

        col_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        # We ignore final_hidden_states here because the reference function ignores it.
        # Compute required shapes
        batch_size = final_hidden_states.shape[0] if final_hidden_states.ndim > 0 else 0
        seq_len = final_hidden_states.shape[1] if final_hidden_states.ndim > 1 else 0
        batch_seq_len = batch_size * seq_len
        hidden_size = final_hidden_states.shape[2] if final_hidden_states.ndim > 2 else 0

        # Number of tokens
        n_tokens = expert_outputs.shape[0]

        # Allocate output (start from zeros to match index_add semantics)
        # Using empty and zero_ ensures robust zero-init and avoids Triton kernel complexity here.
        out = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=expert_outputs.device)
        out.zero_()

        # Triton requires int32 indices
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Launch Triton kernel: one program per token
        grid = (n_tokens,)
        BLOCK_SIZE = 128  # tuneable: 128 or 256 are reasonable defaults
        scatter_add_atomic_kernel[grid](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune for your GPU
        )

        return out


def run(*args):
    return ModelNew()(*args)
