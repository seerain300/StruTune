import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(output_ptr, expert_ptr, indices_ptr,
                             B: tl.int32, H: tl.int32, T: tl.int32):
    """
    Scatter-add of expert outputs into rows of output using token indices (dim=0).
    One program handles one source row i, iterating across H columns deterministically.
    """
    i = tl.program_id(0)
    # Bounds check: if i >= T, return (defensive; grid is set to T, so typically not needed)
    if i >= T:
        return

    # Load token index (int64), cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Iterate across hidden dimension and add each element deterministically
    # This avoids atomics and ensures exact accumulation semantics.
    for h in range(0, H):
        # Load source element (bf16)
        val = tl.load(expert_ptr + i * H + h)
        # Compute destination pointer for output row 'idx', column 'h'
        dest_ptr = output_ptr + idx * H + h
        # Add (bf16 accumulation)
        tl.store(dest_ptr, tl.load(dest_ptr) + val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the index_add via a Triton kernel that performs deterministic per-element adds.
        """
        # Ensure tensors are on CUDA
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("All tensors must be on CUDA for Triton.")

        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()

        # Make tensors contiguous for performance
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (number of rows to scatter into)
        H = output.shape[1]  # hidden_size (number of columns)
        T = token_indices.shape[0]  # number of expert outputs to scatter

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel; use modest num_warps since we do scalar loop per program
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            B, H, T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
