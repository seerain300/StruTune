import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def scatter_add_rows_per_column_kernel(
    output_ptr,          # *bf16, shape [B, H]
    expert_ptr,          # *bf16, shape [T, H]
    indices_ptr,         # *i64,  shape [T]
    B: tl.int32,         # batch_seq_len (rows in output)
    H: tl.int32,         # hidden_size (columns)
    T: tl.int32,         # number of expert outputs
):
    # One program per source row i
    i = tl.program_id(0)

    # Load token index for this source row (int64 -> int32 for pointer arithmetic)
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Bounds guard: i must be in [0, T), idx in [0, B)
    # (Inputs are assumed valid by caller; guards for robustness)
    if (i >= T) or (idx < 0) or (idx >= B):
        return

    # Iterate over hidden columns one-by-one to ensure deterministic semantics and avoid vectorization issues
    for h in range(0, H):
        # Load expert output value for this row and column (bf16)
        val = tl.load(expert_ptr + i * H + h)  # bf16
        # Load current value from output at (idx, h)
        current = tl.load(output_ptr + idx * H + h)  # bf16
        # Add and store back
        tl.store(output_ptr + idx * H + h, current + val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This kernel performs scatter-add along dim=0 without atomics, iterating per hidden column
        to minimize numerical discrepancies and ensure correctness.
        """
        # Fallback to PyTorch if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure contiguity
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Launch Triton kernel
        scatter_add_rows_per_column_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
