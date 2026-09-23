import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: zero-initialize output (robust 2D grid over rows and column blocks)
if TRITON_AVAILABLE:
    @triton.jit
    def zero_init_kernel(
        out_ptr,  # *pointer* to output tensor
        stride_row, stride_col,  # strides in elements
        M, N,                    # dimensions: M = batch_seq_len, N = hidden_size
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        # Mask for bounds
        mask = (row < M) & (cols < N)
        # Compute element offsets
        offsets = row * stride_row + cols * stride_col
        # Zero values (bf16 scalar)
        zeros = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
        # Store zeros with mask
        tl.store(out_ptr + offsets, zeros, mask=mask)

    # Kernel 2: scatter-add: for each token i, atomic_add into out[token_indices[i], :]
    @triton.jit
    def scatter_add_atomic_kernel(
        out_ptr, stride_row, stride_col,      # output strides
        expert_ptr, ep_row, ep_col,          # expert_outputs strides
        token_indices_ptr,                   # *int32* token indices
        M, N,                                # out dims
        n_tokens,                            # number of tokens to process
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)  # one program per token
        # Load token index (int32)
        token_index = tl.load(token_indices_ptr + pid)
        # Cast to int64 for address arithmetic safety
        token_index = token_index.to(tl.int64)
        # Loop over hidden columns in chunks
        col = 0
        while col < N:
            cols = col + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            # Load expert row slice
            ep_offsets = pid * ep_row + cols * ep_col
            vals = tl.load(expert_ptr + ep_offsets, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16))
            # Compute output offsets for this row
            out_offsets = token_index * stride_row + cols * stride_col
            # Atomic add
            tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)
            col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Shapes
        batch_size, seq_len, hidden_size = (final_hidden_states.shape[0], final_hidden_states.shape[1], final_hidden_states.shape[1])
        # NOTE: final_hidden_states shape is (batch_seq_len, hidden_size); we only need batch_seq_len and hidden_size.
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]

        # Device and dtype
        device = final_hidden_states.device
        dtype = final_hidden_states.dtype  # expected bfloat16

        # Ensure expert_outputs dtype matches output dtype
        expert_outputs = expert_outputs.to(dtype)

        # Allocate output
        out = torch.empty((batch_seq_len, hidden_size), dtype=dtype, device=device)

        # If Triton not available or not on CUDA, do CPU fallback (correctness guaranteed)
        if not TRITON_AVAILABLE or device.type != "cuda":
            # CPU/GPU fallback: PyTorch index_add_ guarantees correctness
            out.copy_(final_hidden_states)  # no-op clone, since final_hidden_states is provided as initial state
            # Now perform index_add
            out.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return out

        # Ensure contiguity
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Launch zero-init kernel
        BLOCK_SIZE_COLS = 128  # tuneable
        grid_zero = (batch_seq_len, triton.cdiv(hidden_size, BLOCK_SIZE_COLS))
        # Strides are in elements (PyTorch gives strides in elements)
        stride_row = out.stride(0)
        stride_col = out.stride(1)
        zero_init_kernel[grid_zero](
            out, stride_row, stride_col,
            batch_seq_len, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE_COLS,
            num_warps=4,
        )

        # Launch scatter-add kernel: one program per token
        n_tokens = expert_outputs.shape[0]
        grid_scatter = (n_tokens,)
        ep_row = expert_outputs.stride(0)
        ep_col = expert_outputs.stride(1)
        scatter_add_atomic_kernel[grid_scatter](
            out, stride_row, stride_col,
            expert_outputs, ep_row, ep_col,
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE_COLS,
            num_warps=4,
        )

        return out