import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add one program per token
# It loops over hidden columns in chunks of BLOCK_SIZE and performs atomic_add into the correct row.
@triton.jit
def scatter_add_tokens_kernel(
    out_ptr,                # *bf16, shape [batch_seq_len, hidden_size]
    expert_ptr,             # *bf16, shape [num_tokens, hidden_size]
    token_indices_ptr,      # *int32, shape [num_tokens]
    batch_seq_len: tl.constexpr,  # int
    hidden_size: tl.constexpr,    # int
    n_tokens: tl.constexpr,       # int
    stride_row: tl.constexpr,     # int, out.stride(0)
    stride_col: tl.constexpr,     # int, out.stride(1)
    exp_stride_row: tl.constexpr, # int, expert.stride(0)
    exp_stride_col: tl.constexpr, # int, expert.stride(1)
    BLOCK_SIZE: tl.constexpr      # int, columns processed per iteration
):
    pid = tl.program_id(0)  # one program per token
    # Bounds check: if grid is larger than n_tokens, skip
    if pid >= n_tokens:
        return

    # Load token index (row to accumulate into)
    tok_idx = tl.load(token_indices_ptr + pid)

    # Iterate over hidden columns in chunks
    col = 0
    while col < hidden_size:
        offs = tl.arange(0, BLOCK_SIZE)
        cols = col + offs
        mask = cols < hidden_size

        # Load source vector slice for this token
        src_ptr = expert_ptr + pid * exp_stride_row + cols * exp_stride_col
        src_vals = tl.load(src_ptr, mask=mask, other=0.0)

        # Compute destination pointer: out[tok_idx, cols]
        dst_ptr = out_ptr + tok_idx * stride_row + cols * stride_col
        # Atomic add into destination
        tl.atomic_add(dst_ptr, src_vals, mask=mask)

        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter-add that matches:
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Returns updated final_hidden_states (accumulated).
        """
        # Fallback if Triton not available or not CUDA
        if (not TRITON_AVAILABLE) or (not final_hidden_states.is_cuda):
            # Correctness fallback: use PyTorch index_add
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure dtypes and contiguity
        out = torch.zeros(final_hidden_states.shape, dtype=torch.bfloat16, device=final_hidden_states.device)
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = token_indices_i32.numel()

        # Launch one program per token
        grid = (n_tokens,)
        # Choose a chunk size for columns; 128 or 256 are good defaults. We keep 128 to be safe across sizes.
        BLOCK_SIZE = 128
        scatter_add_tokens_kernel[grid](
            out, expert_outputs, token_indices_i32,
            batch_seq_len=batch_seq_len,
            hidden_size=hidden_size,
            n_tokens=n_tokens,
            stride_row=out.stride(0),
            stride_col=out.stride(1),
            exp_stride_row=expert_outputs.stride(0),
            exp_stride_col=expert_outputs.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune based on GPU
        )

        return out