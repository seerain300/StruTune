import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add in chunks for better vectorization
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

        # Target row index for this element
        idx = tl.load(idx_ptr + pid)  # int32

        # Iterate over hidden_size in chunks of BLOCK_SIZE
        j = 0
        while j < n_cols:
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols

            # Load the chunk of the expert vector (bf16)
            vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)

            # Compute output pointers for this row and chunk, then atomic add
            out_row_ptr = out_ptr + idx * n_cols + cols
            tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)

            j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate inputs
        if final_hidden_states.dim() != 2:
            raise ValueError("final_hidden_states must be 2D (n_rows, hidden_size)")
        if expert_outputs.dim() != 2:
            raise ValueError("expert_outputs must be 2D (n_indices, hidden_size)")
        if token_indices.dim() != 1:
            raise ValueError("token_indices must be 1D (n_indices,)")

        # If Triton/CUDA not available, fallback to PyTorch for correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            # Ensure dtype/device compatibility for index_add
            output.index_add_(0, token_indices.to(output.device), expert_outputs.to(output.dtype))
            return output

        # Prepare output (clone to match original semantics)
        output = final_hidden_states.clone()

        # Ensure dtypes and device
        if output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Shapes
        n_rows, n_cols = output.shape
        n_indices = token_indices.numel()

        # Choose BLOCK_SIZE: power-of-two up to 256
        # Use the largest power-of-two <= min(256, n_cols)
        def largest_pow2_leq(x: int) -> int:
            # Ensure at least 1
            if x <= 1:
                return 1
            # Compute largest power-of-two <= x
            return 1 << (x.bit_length() - 1)

        BLOCK_SIZE = min(256, largest_pow2_leq(int(n_cols)))

        # Launch Triton kernel: one program per index
        grid = (n_indices,)
        scatter_add_experts_kernel[grid](
            output,               # out_ptr
            token_indices,        # idx_ptr
            expert_outputs,       # vec_ptr
            n_indices,            # n_indices
            n_cols,               # n_cols
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
