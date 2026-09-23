import torch

# Triton is required; define the kernel we will actually use.
try:
    import triton
    import triton.language as tl
except Exception:
    raise RuntimeError("Triton is required for this implementation.")

@triton.jit
def scatter_add_rows_atomic_kernel(
    output_ptr,          # *ptr to output tensor [B, H], dtype bfloat16
    expert_ptr,          # *ptr to expert_outputs tensor [T, H], dtype bfloat16
    indices_ptr,         # *ptr to token_indices tensor [T], dtype int64
    B: tl.int32,         # batch_seq_len (rows in output)
    H: tl.int32,         # hidden_size (cols)
    T: tl.int32,         # number of expert outputs
):
    # Each program handles one source row i
    i = tl.program_id(axis=0)
    if i >= T:
        return

    # Load the target row index (int64), cast to int32 for address arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Vector of column offsets [0..H)
    offs = tl.arange(0, H)

    # Compute pointers for this row: output[idx, offs] and expert[i, offs]
    out_ptrs = output_ptr + idx * H + offs
    exp_ptrs = expert_ptr + i * H + offs

    # Load the row vector from expert_outputs (bfloat16)
    v = tl.load(exp_ptrs)

    # Atomic add into output row
    tl.atomic_add(out_ptrs, v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along rows:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are on CUDA for Triton execution
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("All inputs must be on CUDA for Triton execution.")

        # Clone to match original behavior
        output = final_hidden_states.clone()
        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel
        scatter_add_rows_atomic_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=4,   # reasonable default for vectorized H up to a few hundred; Triton will handle scheduling
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
