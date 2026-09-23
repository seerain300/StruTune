import torch
import triton
import triton.language as tl


@triton.jit
def per_token_scatter_add_kernel(
    out_ptr,               # *const half, shape (batch_seq_len, hidden_size)
    expert_ptr,            # *const half, shape (num_selected_tokens, hidden_size)
    token_indices_ptr,     # *const int32, shape (num_selected_tokens,)
    stride_out_row,        # int32
    stride_out_col,        # int32
    stride_exp_row,        # int32
    stride_exp_col,        # int32
    batch_seq_len,         # int32
    hidden_size,           # int32
    n_tokens,              # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per token
    # Bounds check on tokens (if grid > n_tokens)
    if pid >= n_tokens:
        return

    # Load token index (int32)
    token_index = tl.load(token_indices_ptr + pid)

    # Prepare column offsets for chunked iteration
    col = 0
    while col < hidden_size:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size

        # Compute pointers for current chunk
        # out row pointer
        out_row_ptr = out_ptr + token_index * stride_out_row + offs * stride_out_col
        # expert row pointer (pid selects token i)
        exp_row_ptr = expert_ptr + pid * stride_exp_row + offs * stride_exp_col

        # Load current out values (assume initialized clone already, keep them)
        out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)
        # Load expert contribution
        exp_vals = tl.load(exp_row_ptr, mask=mask, other=0.0)
        # Add contribution
        out_vals += exp_vals
        # Store back
        tl.store(out_row_ptr, out_vals, mask=mask)

        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized replacement for run:
        - Preserves initial random values by cloning final_hidden_states.
        - Performs per-token scatter-add in Triton.
        """
        # Ensure tensors are on CUDA (Triton requires CUDA)
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device."

        # Clone to preserve initial random values (matches original behavior)
        out = final_hidden_states.clone()

        # Ensure contiguity
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Number of tokens
        n_tokens = token_indices.numel()
        batch_seq_len = out.shape[0]
        hidden_size = out.shape[1]

        # Triton expects int32 indices for arithmetic
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose chunk size for columns
        BLOCK_SIZE = 128  # tuneable: 128 or 256 depending on GPU and hidden_size

        # Launch one program per token
        grid = (n_tokens,)
        per_token_scatter_add_kernel[grid](
            out, expert_outputs, token_indices_i32,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
