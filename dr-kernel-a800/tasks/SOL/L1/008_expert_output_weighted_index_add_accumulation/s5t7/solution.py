import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_experts_kernel(
        out_ptr,          # *bf16, shape (n_rows, n_cols)
        idx_ptr,          # *i32, shape (n_indices,)
        vec_ptr,          # *bf16, shape (n_indices, n_cols)
        n_indices: tl.int32,
        n_cols: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        if pid >= n_indices:
            return

        # Load target row index for this element
        idx = tl.load(idx_ptr + pid)  # int32

        # Iterate over columns in chunks of BLOCK_SIZE
        j = 0
        while j < n_cols:
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols

            # Load vector chunk (bf16)
            vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)

            # Atomic add into the output row at row idx
            out_row_ptr = out_ptr + idx * n_cols + cols
            tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)

            j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # If Triton/CUDA not available, fallback to PyTorch for correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            # PyTorch index_add semantics (dim=0, index tensor)
            output.index_add_(0, token_indices.to(output.device), expert_outputs.to(output.dtype))
            return output

        # Prepare output and inputs
        output = final_hidden_states.clone()
        # Triton prefers int32 for indices
        idx32 = token_indices.to(torch.int32)
        # Ensure contiguous tensors for predictable strides
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        n_indices = idx32.numel()
        n_rows, n_cols = output.shape

        # Choose a power-of-two BLOCK_SIZE up to 512 for better vectorization
        def _next_power_of_two(x: int) -> int:
            if x <= 64:
                return 64
            return 1 << ((x - 1).bit_length())

        BLOCK_SIZE = min(512, _next_power_of_two(n_cols))

        # Launch Triton kernel: one program per index
        grid = (n_indices,)
        scatter_add_experts_kernel[grid](
            output, idx32, expert_outputs,
            n_indices, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
